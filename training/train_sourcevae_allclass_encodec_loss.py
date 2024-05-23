import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from models.env import AttrDict, build_env
from utils import mel_spectrogram
from models.msstftd import MultiScaleSTFTDiscriminator

from models.SourceVAEAllClass import SourceVAE

from utils import plot_spectrogram
from utils import scan_checkpoint
from utils import load_checkpoint
from utils import save_checkpoint

# from data.dataset_musdb import dataset_musdb, collate_func_musdb
# from data.dataset_vocal import dataset_vocal, collate_func_vocals
from data.dataset_musdb_sourcevae import dataset_musdb_sourcevae, collate_funcs_sourcevae

from losses import total_loss, disc_loss

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    torch.cuda.set_device(rank)
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    sourceVAE = SourceVAE(h).to(device)
 
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)
    if rank == 0:
        print(sourceVAE)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        sourceVAE.load_state_dict(state_dict_g['sourcevae'])

        mstftd.load_state_dict(state_dict_do['mstftd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']


    if h.num_gpus > 1:
        sourceVAE = DistributedDataParallel(
            sourceVAE, device_ids=[rank]).to(device)
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(
        itertools.chain(sourceVAE.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(
        itertools.chain(mstftd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    trainset = dataset_musdb_sourcevae(
        musdb_root=a.musdb_root,
        sample_rate=16000,
        mode='train',
        source_types=['vocals', 'drums', 'bass', 'other'],
        seconds=h.seconds,
        len_ds=5000
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    collate_func = collate_funcs_sourcevae
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_func
        )

    if rank == 0:
        validset = dataset_musdb_sourcevae(
            musdb_root=a.musdb_root,
            sample_rate=16000,
            mode='train',
            source_types=['vocals', 'drums', 'bass', 'other'],
            seconds=h.seconds,
            len_ds=100
        )
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_func)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    plot_gt_once = False
    sourceVAE.train()
    mstftd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            # if i > 10:
            #     break
            if rank == 0:
                start_b = time.time()
            for key in batch.keys():
                if type(batch[key])==torch.Tensor:
                    batch[key] = torch.autograd.Variable(batch[key].to(device, non_blocking=True))

            batch = sourceVAE(batch)
            y_g_hat = batch['output'] # 4*bs, 1, T
            y = batch['reference'] # 4*bs, 1, T

            optim_d.zero_grad()

            # mstft
            logits_real, _ = mstftd(y)
            logits_fake, _ = mstftd(y_g_hat.detach())
            loss_disc_all = disc_loss(logits_real, logits_fake) # compute discriminator loss

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()


            logits_real, fmap_real = mstftd(y)
            logits_fake, fmap_fake = mstftd(y_g_hat)
            
            loss_g, l_t, l_f = total_loss(fmap_real, logits_fake, fmap_fake, y, y_g_hat)

            loss_kl = h.lambda_kl * batch['loss_KLD'].mean()
            # loss_l1 = 5 * batch['loss_l1']
            # loss_snr = 10 * batch['loss_snr'].mean()
            loss_gen_all = loss_g + loss_kl

            loss_gen_all.backward()
            optim_g.step()
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                        
                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, loss kl : {:4.3f}, loss time : {:4.3f}, loss frequency : {:4.3f}, snr : {:4.3f}, s/b : {:4.3f}'.
                        format(steps, loss_gen_all.item(), loss_kl.item(), l_t.item(), l_f.item(), -batch['loss_snr'].mean().item(), time.time() - start_b))
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path,
                                                           steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'sourcevae': (sourceVAE.module if h.num_gpus > 1
                                          else sourceVAE).state_dict()
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mstftd': (mstftd.module
                                       if h.num_gpus > 1 else mstftd).state_dict(),
                            'optim_g':
                            optim_g.state_dict(),
                            'optim_d':
                            optim_d.state_dict(),
                            'steps':
                            steps,
                            'epoch':
                            epoch
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                # if True:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/loss_kl", loss_kl, steps)
                    sw.add_scalar("training/loss time", l_t.item(), steps)
                    sw.add_scalar("training/loss frequency", l_f.item(), steps)
                    sw.add_scalar("training/snr", -batch['loss_snr'].mean().item(), steps)

                # Validation
                if steps % a.validation_interval == 0:
                # if True:
                    sourceVAE.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    val_kl_tot = 0
                    val_l1_tot = 0
                    val_snr_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            if j > 40:
                                break
                            for key in batch.keys():
                                if type(batch[key])==torch.Tensor:
                                    batch[key] = torch.autograd.Variable(batch[key].to(device, non_blocking=True))
                            batch = sourceVAE(batch)
                            y_g_hat = batch['output'] # bs, 1, T
                            y = batch['reference'] # bs, 1, T # bs, 1, T
                            
                            y_mel = mel_spectrogram(
                                y.squeeze(1), h.n_fft, h.num_mels,
                                h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                                h.fmax_for_loss)
                            y_g_hat_mel = mel_spectrogram(
                                y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                                h.fmax_for_loss)
                            i_size = min(y_mel.size(2), y_g_hat_mel.size(2))
                            val_err_tot += F.l1_loss(
                                y_mel[:, :, :i_size],
                                y_g_hat_mel[:, :, :i_size]).item()
                            val_kl_tot += batch['loss_KLD'].clone().mean().item()
                            val_l1_tot += batch['loss_l1'].clone().item()
                            val_snr_tot += (-batch['loss_snr'].clone().mean().item())

                            if j <= 8:
                                # if steps == 0:
                                # if not plot_gt_once:
                                sw.add_audio('gt_vocals/y_{}'.format(j), y[0],
                                                steps, h.sampling_rate)
                                sw.add_audio('generated_vocals/y_hat_{}'.format(j),
                                             y_g_hat[0], steps, h.sampling_rate)
                                sw.add_audio('gt_drums/y_{}'.format(j), y[1],
                                                steps, h.sampling_rate)
                                sw.add_audio('generated_drums/y_hat_{}'.format(j),
                                             y_g_hat[1], steps, h.sampling_rate)
                                sw.add_audio('gt_bass/y_{}'.format(j), y[2],
                                                steps, h.sampling_rate)
                                sw.add_audio('generated_bass/y_hat_{}'.format(j),
                                             y_g_hat[2], steps, h.sampling_rate)
                                sw.add_audio('gt_other/y_{}'.format(j), y[3],
                                                steps, h.sampling_rate)
                                sw.add_audio('generated_other/y_hat_{}'.format(j),
                                             y_g_hat[3], steps, h.sampling_rate)


                        val_err = val_err_tot / (j + 1)
                        val_kl = val_kl_tot / (j + 1)
                        val_l1 = val_l1_tot / (j + 1)
                        val_snr = val_snr_tot / (j + 1)

                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        sw.add_scalar("validation/kl_divergence", val_kl, steps)
                        sw.add_scalar("validation/l1_loss", val_l1, steps)
                        sw.add_scalar("validation/snr", val_snr, steps)
                        if not plot_gt_once:
                            plot_gt_once = True

                    sourceVAE.train()

            steps += 1
        scheduler_g.step()
        # if steps < a.vanilla_steps and h.codec_loss:
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(
                epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    # parser.add_argument('--group_name', default=None)
    # parser.add_argument('--input_wavs_dir', default='../datasets/audios')
    parser.add_argument('--musdb_root', required=True)
    parser.add_argument('--vocalset_root', required=True)
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=10000, type=int)
    parser.add_argument('--vanilla_steps', default=10000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=1, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, ))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()