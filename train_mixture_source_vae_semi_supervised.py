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
# from models.SourceVAE import SourceVAE
# from models.SourceVAESubband2 import SourceVAE
# from models.MixtureVAESingle import MixtureVAE
from models.MixtureVAEFull import MixtureVAEFull
from models.models_hificodec import MultiPeriodDiscriminator
from models.models_hificodec import MultiScaleDiscriminator
from models.models_hificodec import feature_loss
from models.models_hificodec import generator_loss
from models.models_hificodec import discriminator_loss
from utils import plot_spectrogram
from utils import scan_checkpoint
from utils import load_checkpoint
from utils import save_checkpoint

from data.dataset_musdb import dataset_musdb, collate_func_musdb
from data.dataset_semi_supervised import dataset_semi_supervised, collate_func_semi_supsevised

torch.backends.cudnn.benchmark = True

from torch.cuda.amp import autocast, GradScaler


# Helper function to select parameters
def get_parameters(model, keyword):
    for name, param in model.named_parameters():
        if keyword in name:
            yield param

def default_mel(y, h):
    if y.shape[1] == 1:
        return mel_spectrogram(
            y.squeeze(1), h.n_fft, h.num_mels,
            h.sampling_rate, h.hop_size, h.win_size, h.fmin,
            h.fmax_for_loss)
    else:
        return mel_spectrogram(
            y, h.n_fft, h.num_mels,
            h.sampling_rate, h.hop_size, h.win_size, h.fmin,
            h.fmax_for_loss)

# def feature_loss(fmap_r, fmap_g, bs):
#     loss1 = 0
#     loss2 = 0
#     for dr, dg in zip(fmap_r, fmap_g):
#         for rl, gl in zip(dr, dg):
#             loss1 += torch.mean(torch.abs(rl[:bs] - gl[:bs]))
#             loss2 += torch.mean(torch.abs(rl[bs:] - gl[bs:]))

#     # return loss * 2
#     return loss1 + loss2


# def generator_loss(disc_outputs, bs):
#     loss = 0
#     gen_losses = []
#     for dg in disc_outputs:
#         l1 = torch.mean((1 - dg[:bs])**2)
#         l2 = torch.mean((1 - dg[bs:])**2)
#         l = l1 + l2
#         gen_losses.append(l)
#         loss = loss + l

#     return loss, gen_losses

# def discriminator_loss(disc_real_outputs, disc_generated_outputs, bs):
#     loss = 0
#     r_losses = []
#     g_losses = []
#     for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
#         r_loss = torch.mean((1 - dr[:bs])**2) + torch.mean((1 - dr[bs:])**2)
#         g_loss = torch.mean(dg[:bs]**2) + torch.mean(dg[bs:]**2)
#         loss += (r_loss + g_loss)
#         r_losses.append(r_loss.item())
#         g_losses.append(g_loss.item())

#     return loss, r_losses, g_losses

def mel_loss(h, y, y_g_hat, bs, supervised=True):
    y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
            h.hop_size, h.win_size, h.fmin,
            h.fmax_for_loss)  # 1024, 80, 24000, 240,1024
    y_mel = mel_spectrogram(
            y.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
            h.hop_size, h.win_size, h.fmin,
            h.fmax_for_loss)  # 1024, 80, 24000, 240,1024
    
    y_r_mel_1 = mel_spectrogram(
        y.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
        h.fmin, h.fmax_for_loss)
    y_g_mel_1 = mel_spectrogram(
        y_g_hat.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512,
        h.fmin, h.fmax_for_loss)
    y_r_mel_2 = mel_spectrogram(
        y.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin,
        h.fmax_for_loss)
    y_g_mel_2 = mel_spectrogram(
        y_g_hat.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256,
        h.fmin, h.fmax_for_loss)


    if supervised:
        loss_mel1_sup = F.l1_loss(y_r_mel_1[:bs], y_g_mel_1[:bs])
        loss_mel2_sup = F.l1_loss(y_r_mel_2[:bs], y_g_mel_2[:bs])
        loss_mel_sup = F.l1_loss(y_mel[:bs], y_g_hat_mel[:bs]) * 45 + loss_mel1_sup + loss_mel2_sup

        loss_mel1_mix = F.l1_loss(y_r_mel_1[bs*1:bs*5], y_g_mel_1[bs*1:bs*5])
        loss_mel2_mix = F.l1_loss(y_r_mel_2[bs*1:bs*5], y_g_mel_2[bs*1:bs*5])
        loss_mel_mix = F.l1_loss(y_mel[bs*1:bs*5], y_g_hat_mel[bs*1:bs*5]) * 45 + loss_mel1_mix + loss_mel2_mix

        loss_mel1_src = F.l1_loss(y_r_mel_1[bs*5:], y_g_mel_1[bs*5:])
        loss_mel2_src = F.l1_loss(y_r_mel_2[bs*5:], y_g_mel_2[bs*5:])
        loss_mel_src = F.l1_loss(y_mel[bs*5:], y_g_hat_mel[bs*5:]) * 45 + loss_mel1_src + loss_mel2_src
        
        return loss_mel_sup, loss_mel_mix*4, loss_mel_src*4
    else:
        loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
        loss_mel2 = F.l1_loss(y_r_mel_2[:bs], y_g_mel_2)
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45 + loss_mel1 + loss_mel2

        return loss_mel

def loss_descriminators(mpd, msd, mstftd, y, y_g_hat, bs):
    # MPD
    y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
        y_df_hat_r, y_df_hat_g)

    # MSD
    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
        y_ds_hat_r, y_ds_hat_g)

    # MSTFTD
    y_disc_r, fmap_r = mstftd(y)
    y_disc_gen, fmap_gen = mstftd(y_g_hat.detach())
    loss_disc_stft, losses_disc_stft_r, losses_disc_stft_g = discriminator_loss(
        y_disc_r, y_disc_gen)
    loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_stft

    return loss_disc_all

def loss_generators(mpd, msd, mstftd, y, y_g_hat, bs):
    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
    y_stftd_hat_r, fmap_stftd_r = mstftd(y)
    y_stftd_hat_g, fmap_stftd_g = mstftd(y_g_hat)
    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
    loss_fm_stft = feature_loss(fmap_stftd_r, fmap_stftd_g)
    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
    loss_gen_stft, losses_gen_stft = generator_loss(y_stftd_hat_g)

    # return loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f
    return loss_gen_s + loss_gen_f + loss_gen_stft + loss_fm_s + loss_fm_f + loss_fm_stft

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

    
    # ckpt_source = torch.load(os.path.join(h.sourcevae_ckpt_dir, 'ckpt_'+h.source_type))
    # ckpt_model_pretrained = torch.load(h.mixvae_ckpt_dir)

    ckpt_vocals = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_vocals'))
    ckpt_drums = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_drums'))
    ckpt_bass = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_bass'))
    ckpt_other = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_other'))
    model_ckpts = {
        'vocals':ckpt_vocals, 'drums':ckpt_drums, 'bass':ckpt_bass, 'other':ckpt_other, 
    }
    mixtureVAE = MixtureVAEFull(h, load_model=True, model_ckpt=model_ckpts).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mstftd = MultiScaleSTFTDiscriminator(32).to(device)

    # load pretrained discriminators, pretrained from posterior only training
    state_dict_do_pretrain = load_checkpoint(h.do_ckpt_path, device)
    mpd.load_state_dict(state_dict_do_pretrain['mpd'])
    msd.load_state_dict(state_dict_do_pretrain['msd'])
    mstftd.load_state_dict(state_dict_do_pretrain['mstftd'])
    print('------------------------------------pretrained disciminator pre-loaded-----------------------------------------')

    if rank == 0:
        print(mixtureVAE)
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
        mixtureVAE.load_state_dict(state_dict_g['mixturevae'])

        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        mstftd.load_state_dict(state_dict_do['mstftd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']


    if h.num_gpus > 1:
        mixtureVAE = DistributedDataParallel(
            mixtureVAE, device_ids=[rank], find_unused_parameters=True).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        mstftd = DistributedDataParallel(mstftd, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(
        itertools.chain(mixtureVAE.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.Adam(
        itertools.chain(msd.parameters(), mpd.parameters(),
                        mstftd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2])

    # scaler_gen=torch.cuda.amp.GradScaler()
    # scaler_disc=torch.cuda.amp.GradScaler()

    # load optim g here for continual training
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    trainset = dataset_semi_supervised(
        musdb_root='/data/romit/alan/musdb18',
        fma_root='/data/romit/alan/fma/fma',
        sample_rate=16000,
        mode='train',
        seconds=h.seconds,
        p=1,
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    collate_func = collate_func_semi_supsevised
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
        validset = dataset_semi_supervised(
            musdb_root='/data/romit/alan/musdb18',
            fma_root='/data/romit/alan/fma/fma',
            sample_rate=16000,
            mode='validation',
            seconds=h.seconds,
            p=1,
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
    mixtureVAE.train()

    mstftd.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            # if i > 10:
            #     break
            if steps % 3 == 0 or steps < a.supervised_steps:
                supervised=True
            else:
                supervised=False
            supervised_keys = ['vocals', 'drums', 'bass', 'other', 'mixture']
            unsupervised_keys = ['mixture_fma']

            if rank == 0:
                start_b = time.time()
            for key in batch.keys():
                if type(batch[key])==torch.Tensor:
                    if supervised and key in supervised_keys:
                        batch[key] = torch.autograd.Variable(batch[key][..., :40000].to(device, non_blocking=True))
                    elif key in unsupervised_keys:
                        batch[key] = torch.autograd.Variable(batch[key].to(device, non_blocking=True))

            if supervised:
                batch = mixtureVAE(batch, sample_posterior=True, unsupervised=False, output_source=False, supervised=True)

                # merge together for all loss calculation
                bs = batch['mixture'].shape[0]
                y = torch.cat([batch['mixture'], 
                            batch['vocals'], 
                            batch['drums'], 
                            batch['bass'], 
                            batch['other'],
                            batch['vocals'], 
                            batch['drums'], 
                            batch['bass'], 
                            batch['other']], dim=0)
                y_g_hat = torch.cat([batch['mixture_dec_mix'], 
                                    batch['vocals_dec_mix'], 
                                    batch['drums_dec_mix'], 
                                    batch['bass_dec_mix'], 
                                    batch['other_dec_mix'],
                                    batch['vocals_dec'], 
                                    batch['drums_dec'], 
                                    batch['bass_dec'], 
                                    batch['other_dec']], dim=0)
            else:
                batch = mixtureVAE(batch, sample_posterior=True, unsupervised=True, output_source=False, supervised=False)

                # merge together for all loss calculation
                bs = batch['mixture_fma'].shape[0]
                y = batch['mixture_fma']
                y_g_hat = batch['dec_mixture_fma']

            optim_d.zero_grad()

            # with autocast():
                # discriminator loss
            loss_disc_all = loss_descriminators(mpd, msd, mstftd, y, y_g_hat, bs)

            loss_disc_all.backward()
            # scaler_disc.scale(loss_disc_all).backward()

            optim_d.step()
            # scaler_disc.step(optim_d)

            # scaler_disc.update()

            # Generator
            optim_g.zero_grad()

            # with autocast():
            if supervised:
                loss_mel_sup, loss_mel_mix, loss_mel_src = mel_loss(h, y, y_g_hat, bs, supervised=True)
                loss_mel_all = loss_mel_sup + loss_mel_mix + loss_mel_src
            else:
                loss_mel_all = mel_loss(h, y, y_g_hat, bs, supervised=False)

            loss_g_all = loss_generators(mpd, msd, mstftd, y, y_g_hat, bs)

            if supervised:

                loss_kl_source = 0
                loss_posterior_matching = 0
                for source_type in ['vocals', 'drums', 'other', 'bass']:
                    loss_kl_source = loss_kl_source + h.lambda_kl * batch[source_type+'_loss_KLD'].mean()
                    loss_posterior_matching = loss_posterior_matching + h.lambda_posterior * batch[source_type+'_loss_posterior_matching'].mean()

                loss_gen_all = loss_g_all + loss_mel_all + loss_kl_source + loss_posterior_matching

            else:
                kl_loss_unsupervised = h.lambda_kl * batch['unsupervised_kl'].mean()
                loss_gen_all = loss_g_all + loss_mel_all + kl_loss_unsupervised

            loss_gen_all.backward()
            # scaler_gen.scale(loss_gen_all).backward()
            optim_g.step()
            # scaler_gen.step(optim_g)
            # scaler_gen.update()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():

                        if supervised:
                            y_mel = default_mel(y[bs:bs*5], h)
                            y_g_hat_mel_mix = default_mel(y_g_hat[bs:bs*5], h)
                            y_g_hat_mel_source = default_mel(y_g_hat[bs*5:], h)
                            y_mel_sup = default_mel(y[:bs], h)
                            y_g_hat_mel_sup = default_mel(y_g_hat[:bs], h)
                        
                            mel_error_source = F.l1_loss(y_mel, y_g_hat_mel_source).item()
                            mel_error_mixture = F.l1_loss(y_mel, y_g_hat_mel_mix).item()
                            mel_error_sup = F.l1_loss(y_mel_sup, y_g_hat_mel_sup).item()
                        else:
                            y_mel_unsup = default_mel(y, h)
                            y_g_hat_mel_unsup = default_mel(y_g_hat, h)
                            mel_error_unsup = F.l1_loss(y_mel_unsup, y_g_hat_mel_unsup).item()

                    if supervised:
                        print(
                            'Steps : {:d}, Gen Loss Total : {:4.3f}, mel_error_sup: {:4.3f}, mel_error_source: {:4.3f}, mel_error_mixture: {:4.3f}, s/b : {:4.3f}'.
                            format(steps, loss_gen_all.item(), mel_error_sup, mel_error_source, mel_error_mixture, time.time() - start_b))
                    else:
                        print(
                            'Steps : {:d}, Gen Loss Total : {:4.3f}, mel_error_unsup: {:4.3f}, s/b : {:4.3f}'.
                            format(steps, loss_gen_all.item(), mel_error_unsup, time.time() - start_b))
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mixturevae': (mixtureVAE.module if h.num_gpus > 1
                                          else mixtureVAE).state_dict()
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'mpd': (mpd.module
                                    if h.num_gpus > 1 else mpd).state_dict(),
                            'msd': (msd.module
                                    if h.num_gpus > 1 else msd).state_dict(),
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
                    if supervised:
                        sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                        sw.add_scalar("training/mel_error_sup", mel_error_sup, steps)
                        sw.add_scalar("training/mel_error_source", mel_error_source, steps)
                        sw.add_scalar("training/mel_error_mixture", mel_error_mixture, steps)
                    else:
                        sw.add_scalar("training/gen_loss_total_unsup", loss_gen_all, steps)
                        sw.add_scalar("training/mel_error_unsup", mel_error_unsup, steps)


                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                # if steps % 10 == 0 and steps != 0:
                    mixtureVAE.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    val_err_self_tot = 0
                    val_unsup_err_tot = 0
                    # val_l1_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            if j > 50:
                                break
                            for key in batch.keys():
                                if type(batch[key])==torch.Tensor:
                                    batch[key] = torch.autograd.Variable(batch[key].to(device, non_blocking=True))

                            batch = mixtureVAE(batch, sample_posterior=True, unsupervised=True, output_source=True, supervised=True)


                            mix_audio = batch['mixture']
                            mixture_fma = batch['mixture_fma']
                            # groundtruth sources
                            sw.add_audio('mixture/y_hat_{}'.format(j), mix_audio, steps, h.sampling_rate)
                            sw.add_audio('unsupervised_mixture/y_hat_{}'.format(j), mix_audio, steps, h.sampling_rate)


                            for source_type in ['vocals', 'drums', 'bass', 'other']:
                                y = batch[source_type]
                                # y_drums = batch['drums']
                                # y_bass = batch['bass']
                                # y_other = batch['other']


                                # mixture separation samples after decoding
                                y_source = batch[source_type+'_dec']
                                y_mix = batch[source_type+'_dec_mix']

                                dec_mixture_fma = batch['dec_mixture_fma']
                                mixture_fma_mel = default_mel(mixture_fma, h)
                                dec_mixture_fma_mel = default_mel(dec_mixture_fma, h)


                                y_mel = default_mel(y, h)
                                y_mix_mel = default_mel(y_mix, h)
                                y_source_mel = default_mel(y_source, h)

                                i_size = min(y_mel.size(2), y_mix_mel.size(2))
                                
                                val_error_current = F.l1_loss(y_mel[:, :, :i_size], y_mix_mel[:, :, :i_size]).item()
                                val_error_original = F.l1_loss(y_mel[:, :, :i_size], y_source_mel[:, :, :i_size]).item()
                                val_error_unsupervised = F.l1_loss(mixture_fma_mel[:, :, :i_size], dec_mixture_fma_mel[:, :, :i_size]).item()

                                val_err_tot += val_error_current
                                val_err_self_tot += val_error_original
                                val_unsup_err_tot += val_error_unsupervised

                                if j <= 8:
                                    sw.add_audio('gt/'+source_type+'y_{}'.format(j), y, steps, h.sampling_rate)
                                    sw.add_audio('mix_generated/'+source_type+'y_hat_{}'.format(j), y_mix, steps, h.sampling_rate)
                                    sw.add_audio('self_generated/'+source_type+'y_hat_{}'.format(j), y_source, steps, h.sampling_rate)
                                    sw.add_audio('unsupervised/'+source_type+'y_hat_{}'.format(j), batch['dec_'+source_type+'_fma'], steps, h.sampling_rate)


                        val_err = val_err_tot / (j + 1)
                        val_err_ori = val_err_self_tot / (j + 1)
                        val_err_unsup = val_unsup_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        sw.add_scalar("validation/mel_spec_error_original", val_err_ori, steps)
                        sw.add_scalar("validation/val_err_unsup", val_err_unsup, steps)


                    mixtureVAE.train()

            steps += 1
        
        if steps >= a.supervised_steps:
            scheduler_g.step()
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
    parser.add_argument('--supervised_steps', default=5000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--num_ckpt_keep', default=5, type=int)
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