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
from models.MixtureVAESingle import MixtureVAE
# from models.MixtureVAEFull import MixtureVAEFull
from models.models_hificodec import MultiPeriodDiscriminator
from models.models_hificodec import MultiScaleDiscriminator
from models.models_hificodec import feature_loss
from models.models_hificodec import generator_loss
from models.models_hificodec import discriminator_loss
from utils import plot_spectrogram
from utils import scan_checkpoint
from utils import load_checkpoint
from utils import save_checkpoint

from data.dataset_semi_supervised import dataset_semi_supervised, collate_func_semi_supsevised
from data.dataset_semi_supervised_v2 import dataset_semi_supervised_v2, collate_func_semi_supsevised_v2

torch.backends.cudnn.benchmark = True



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


def mel_loss(h, y, y_g_hat):
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


    loss_mel1 = F.l1_loss(y_r_mel_1, y_g_mel_1)
    loss_mel2 = F.l1_loss(y_r_mel_2, y_g_mel_2)
    #print('loss_mel1, loss_mel2 ', loss_mel1, loss_mel2)
    loss_mel = F.l1_loss(y_mel,
                        y_g_hat_mel) * 45 + loss_mel1 + loss_mel2

    return loss_mel

def loss_descriminators(mpd, msd, mstftd, y, y_g_hat):
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

def loss_generators(mpd, msd, mstftd, y, y_g_hat):
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
    return loss_gen_s, loss_gen_f, loss_gen_stft, loss_fm_s, loss_fm_f, loss_fm_stft

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

    source_types = ['vocals', 'drums', 'bass', 'other']

    ckpt_vocals = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_vocals'))
    ckpt_drums = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_drums'))
    ckpt_bass = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_bass'))
    ckpt_other = torch.load(os.path.join(h.mixvae_ckpt_dir, 'ckpt_other'))
    model_ckpts = {
        'vocals':ckpt_vocals, 'drums':ckpt_drums, 'bass':ckpt_bass, 'other':ckpt_other, 
    }
    mixtureVAEs = {}
    for source_type in source_types:
       mixtureVAEs[source_type] = MixtureVAE(h, load_model=True, model_ckpt=model_ckpts[source_type], load_source_vae=False).to(device)
       mixtureVAEs[source_type].source_type = source_type

    mpds = {}
    msds = {}
    mstftds = {}

    for source_type in source_types+['mixture']:
        mpds[source_type] = MultiPeriodDiscriminator().to(device)
        msds[source_type] = MultiScaleDiscriminator().to(device)
        mstftds[source_type] = MultiScaleSTFTDiscriminator(32).to(device)

        state_dict_do_pretrain = load_checkpoint(h.do_ckpt_path+'/do_'+source_type, device)
        mpds[source_type].load_state_dict(state_dict_do_pretrain['mpd'])
        msds[source_type].load_state_dict(state_dict_do_pretrain['msd'])
        mstftds[source_type].load_state_dict(state_dict_do_pretrain['mstftd'])

    print('------------------------------------pretrained class-wise disciminator pre-loaded-----------------------------------------')

    exp_continue = True

    if rank == 0:
        print(mixtureVAEs['vocals'])
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
        for source_type in source_types:
            mixtureVAEs[source_type].load_state_dict(state_dict_g[source_type])
        for source_type in source_types + ['mixture']:
            mpds[source_type].load_state_dict(state_dict_do[source_type+'_mpd'])
            msds[source_type].load_state_dict(state_dict_do[source_type+'_msd'])
            mstftds[source_type].load_state_dict(state_dict_do[source_type+'_mstftd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        del state_dict_g
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print('finish loading ckpt--------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------')


    if h.num_gpus > 1:
        for source_type in source_types:
            mixtureVAEs[source_type] = DistributedDataParallel(
                mixtureVAEs[source_type], device_ids=[rank], find_unused_parameters=True).to(device)
        for source_type in source_types+['mixture']:
            mpds[source_type] = DistributedDataParallel(mpds[source_type], device_ids=[rank]).to(device)
            msds[source_type] = DistributedDataParallel(msds[source_type], device_ids=[rank]).to(device)
            mstftds[source_type] = DistributedDataParallel(mstftds[source_type], device_ids=[rank]).to(device)

    optim_gs = {}
    optim_g_mixvae = {}
    for source_type in source_types:
        optim_gs[source_type] = torch.optim.Adam(
            itertools.chain(mixtureVAEs[source_type].parameters()),
            h.learning_rate,
            betas=[h.adam_b1, h.adam_b2])
        optim_g_mixvae[source_type] = torch.optim.Adam(
                itertools.chain(get_parameters(mixtureVAEs[source_type], 'MixEncoder')),
                h.learning_rate,
                betas=[h.adam_b1, h.adam_b2])

    optim_ds = {}
    for source_type in source_types+['mixture']:
        optim_ds[source_type] = torch.optim.Adam(
            itertools.chain(msds[source_type].parameters(), mpds[source_type].parameters(),
                            mstftds[source_type].parameters()),
            h.learning_rate,
            betas=[h.adam_b1, h.adam_b2])

    # load optim g here for continual training
    if state_dict_do is not None:
        for source_type in source_types:
            optim_gs[source_type].load_state_dict(state_dict_do[source_type+'_optim_g'])
            optim_g_mixvae[source_type].load_state_dict(state_dict_do[source_type+'_optim_g_mixvae'])
            optim_ds[source_type].load_state_dict(state_dict_do[source_type+'_optim_d'])
        optim_ds['mixture'].load_state_dict(state_dict_do['mixture'+'_optim_d'])


    scheduler_gs = {}
    scheduler_g_mixvae = {}
    scheduler_ds = {}
    for source_type in source_types:
        scheduler_gs[source_type] = torch.optim.lr_scheduler.ExponentialLR(
            optim_gs[source_type], gamma=h.lr_decay, last_epoch=int(steps % a.lr_decay_step)-1)
        scheduler_g_mixvae[source_type] = torch.optim.lr_scheduler.ExponentialLR(
            optim_g_mixvae[source_type], gamma=h.lr_decay, last_epoch=int(steps % a.lr_decay_step)-1)
    for source_type in source_types+['mixture']:
        scheduler_ds[source_type] = torch.optim.lr_scheduler.ExponentialLR(
            optim_ds[source_type], gamma=h.lr_decay, last_epoch=int(steps % a.lr_decay_step)-1)

    trainset = dataset_semi_supervised_v2(
        musdb_root='/data/romit/alan/musdb18',
        fma_root='/data/romit/alan/fma/fma',
        sample_rate=16000,
        mode='train',
        seconds=h.seconds,
        source_types=source_types
    )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    collate_func = collate_func_semi_supsevised_v2
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
    for source_type in source_types:
        mixtureVAEs[source_type].train()

        mstftds[source_type].train()
        mpds[source_type].train()
        msds[source_type].train()

    del state_dict_do
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            # if i > 10:
            #     break
            if steps < 30000:
                unsupervised_steps_per_supervised = 2
            # elif steps < 30000:
            #     unsupervised_steps_per_supervised = 3
            else:
                unsupervised_steps_per_supervised = 3

            if (steps % unsupervised_steps_per_supervised == 0) or (steps < a.supervised_steps):
                supervised=True
            else:
                supervised=False
            supervised_keys = ['vocals', 'drums', 'bass', 'other', 'vocals_mixture', 'drums_mixture', 'bass_mixture', 'other_mixture']
            unsupervised_keys = ['mixture_fma']

            if rank == 0:
                start_b = time.time()
            for key in batch.keys():
                if type(batch[key])==torch.Tensor:
                    if supervised and key in supervised_keys:
                        batch[key] = torch.autograd.Variable(batch[key].to(device, non_blocking=True))
                    elif key in unsupervised_keys:
                        batch[key] = torch.autograd.Variable(batch[key].to(device, non_blocking=True))

            if supervised:
                bs = batch['mixture_fma'].shape[0]
                for source_type in source_types:
                # for source_type in ['drums']:
                
                    new_batch = {}
                    # new_batch[source_type+'_dec'] = batch[source_type+'_dec'].clones()
                    new_batch['mixture'] = batch[source_type+'_mixture'].clone()
                    new_batch[source_type] = batch[source_type].clone()

                    new_batch = mixtureVAEs[source_type](new_batch, separate=False, sample_posterior=True, decode=True, train_source_decoder=True, train_source_encoder=True)
                    # batch = mixtureVAEs[source_type](batch, sample_posterior=True, decode=True, train_source_decoder=True, train_source_encoder=True)

                    # copy back result to batch
                    for key in new_batch.keys():
                        if key not in batch.keys() and key != 'mixture':
                            batch[key] = new_batch[key]

                    y_source = new_batch[source_type+'_dec'] # bs, 1, T
                    y_mix = new_batch[source_type+'_dec_mix'] # bs, 1, T
                    y = new_batch[source_type] # bs, 1, T

                    y_g_hat = torch.cat([y_source, y_mix], dim=0)
                    y = torch.cat([y, y], dim=0)

                    optim_ds[source_type].zero_grad()

                    loss_disc_all = loss_descriminators(mpds[source_type], msds[source_type], mstftds[source_type], y, y_g_hat)

                    loss_disc_all.backward()

                    optim_ds[source_type].step()

                    optim_gs[source_type].zero_grad()

                    loss_mel = mel_loss(h, y, y_g_hat)

                    loss_gen_s, loss_gen_f, loss_gen_stft, loss_fm_s, loss_fm_f, loss_fm_stft = loss_generators(mpds[source_type], msds[source_type], mstftds[source_type], y, y_g_hat)

                    loss_kl_source = h.lambda_kl * new_batch[source_type+'_loss_KLD'].mean()
                    loss_posterior_matching = h.lambda_posterior * new_batch[source_type+'_loss_posterior_matching'].mean()

                    loss_gen_all = loss_gen_s + loss_gen_f + loss_gen_stft + loss_fm_s + loss_fm_f + loss_fm_stft + loss_mel + loss_kl_source + loss_posterior_matching

                    loss_gen_all.backward()

                    optim_gs[source_type].step()

            else:
                for source_type in source_types:
                    batch = mixtureVAEs[source_type](batch, separate=True, sample_posterior=True)
                batch['mixture_dec_fma'] = batch['vocals_dec_fma'] + batch['drums_dec_fma'] + batch['bass_dec_fma'] + batch['other_dec_fma']

                # merge together for all loss calculation
                bs = batch['mixture_fma'].shape[0]
                y = batch['mixture_fma']
                y_g_hat = batch['mixture_dec_fma']

                optim_ds['mixture'].zero_grad()

                loss_disc_all = loss_descriminators(mpds['mixture'], msds['mixture'], mstftds['mixture'], y, y_g_hat)

                loss_disc_all.backward()

                optim_ds['mixture'].step()

                for source_type in source_types:
                    optim_g_mixvae[source_type].zero_grad()

                loss_mel = mel_loss(h, y, y_g_hat)

                loss_gen_s, loss_gen_f, loss_gen_stft, loss_fm_s, loss_fm_f, loss_fm_stft = loss_generators(mpds['mixture'], msds['mixture'], mstftds['mixture'], y, y_g_hat)

                kl_loss_unsupervised = 0
                for source_type in source_types:
                    kl_loss_unsupervised = kl_loss_unsupervised + batch[source_type+'_unsupervised_kl'].mean()
                kl_loss_unsupervised = h.lambda_kl * kl_loss_unsupervised

                loss_gen_all = loss_gen_s + loss_gen_f + loss_gen_stft + loss_fm_s + loss_fm_f + loss_fm_stft + loss_mel + kl_loss_unsupervised

                loss_gen_all.backward()
                for source_type in source_types:
                    optim_g_mixvae[source_type].step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        
                        if supervised:

                            yz = torch.cat([batch['vocals'], 
                                batch['drums'], 
                                batch['bass'], 
                                batch['other'],], dim=0)
                            yz_g_hat = torch.cat([batch['vocals_dec_mix'], 
                                batch['drums_dec_mix'], 
                                batch['bass_dec_mix'], 
                                batch['other_dec_mix'],
                                batch['vocals_dec'], 
                                batch['drums_dec'], 
                                batch['bass_dec'], 
                                batch['other_dec']], dim=0)

                            y_mel = default_mel(yz, h)
                            y_g_hat_mel_mix = default_mel(yz_g_hat[:bs*4], h)
                            y_g_hat_mel_source = default_mel(yz_g_hat[bs*4:], h)
                        
                            mel_error_source = F.l1_loss(y_mel, y_g_hat_mel_source).item()
                            mel_error_mixture = F.l1_loss(y_mel, y_g_hat_mel_mix).item()


                            # yz = batch['vocals']
                            # yz_g_hat = torch.cat([batch['vocals_dec_mix'], batch['vocals_dec']], dim=0)
                            # y_mel = default_mel(yz, h)
                            # y_g_hat_mel_mix = default_mel(yz_g_hat[:bs], h)
                            # y_g_hat_mel_source = default_mel(yz_g_hat[bs:], h)
                            # mel_error_source = F.l1_loss(y_mel, y_g_hat_mel_source).item()
                            # mel_error_mixture = F.l1_loss(y_mel, y_g_hat_mel_mix).item()
                            # print('vocals: ', mel_error_mixture)

                            # yz = batch['drums']
                            # yz_g_hat = torch.cat([batch['drums_dec_mix'], batch['drums_dec']], dim=0)
                            # y_mel = default_mel(yz, h)
                            # y_g_hat_mel_mix = default_mel(yz_g_hat[:bs], h)
                            # y_g_hat_mel_source = default_mel(yz_g_hat[bs:], h)
                            # mel_error_source = F.l1_loss(y_mel, y_g_hat_mel_source).item()
                            # mel_error_mixture = F.l1_loss(y_mel, y_g_hat_mel_mix).item()
                            # print('drums: ', mel_error_mixture)

                            # yz = batch['bass']
                            # yz_g_hat = torch.cat([batch['bass_dec_mix'], batch['bass_dec']], dim=0)
                            # y_mel = default_mel(yz, h)
                            # y_g_hat_mel_mix = default_mel(yz_g_hat[:bs], h)
                            # y_g_hat_mel_source = default_mel(yz_g_hat[bs:], h)
                            # mel_error_source = F.l1_loss(y_mel, y_g_hat_mel_source).item()
                            # mel_error_mixture = F.l1_loss(y_mel, y_g_hat_mel_mix).item()
                            # print('bass: ', mel_error_mixture)

                            # yz = batch['other']
                            # yz_g_hat = torch.cat([batch['other_dec_mix'], batch['other_dec']], dim=0)
                            # y_mel = default_mel(yz, h)
                            # y_g_hat_mel_mix = default_mel(yz_g_hat[:bs], h)
                            # y_g_hat_mel_source = default_mel(yz_g_hat[bs:], h)
                            # mel_error_source = F.l1_loss(y_mel, y_g_hat_mel_source).item()
                            # mel_error_mixture = F.l1_loss(y_mel, y_g_hat_mel_mix).item()
                            # print('other: ', mel_error_mixture)
                        else:
                            yz = batch['mixture_fma']
                            yz_g_hat = batch['mixture_dec_fma']
                            y_mel_unsup = default_mel(yz, h)
                            y_g_hat_mel_unsup = default_mel(yz_g_hat, h)
                            mel_error_unsup = F.l1_loss(y_mel_unsup, y_g_hat_mel_unsup).item()

                    if supervised:
                        print(
                            'Steps : {:d}, Gen Loss Total : {:4.3f}, mel_error_source: {:4.3f}, mel_error_mixture: {:4.3f}, s/b : {:4.3f}'.
                            format(steps, loss_gen_all.item(), mel_error_source, mel_error_mixture, time.time() - start_b))
                    else:
                        print(
                            'Steps : {:d}, Gen Loss Total : {:4.3f}, mel_error_unsup: {:4.3f}, s/b : {:4.3f}'.
                            format(steps, loss_gen_all.item(), mel_error_unsup, time.time() - start_b))
                # checkpointing
                # if steps % a.checkpoint_interval == 0 and steps != 0:
                if False:
                # if steps % 20 == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'vocals': (mixtureVAEs['vocals'].module if h.num_gpus > 1
                                          else mixtureVAEs['vocals']).state_dict(),
                            'drums': (mixtureVAEs['drums'].module if h.num_gpus > 1
                                          else mixtureVAEs['drums']).state_dict(),
                            'bass': (mixtureVAEs['bass'].module if h.num_gpus > 1
                                          else mixtureVAEs['bass']).state_dict(),
                            'other': (mixtureVAEs['other'].module if h.num_gpus > 1
                                          else mixtureVAEs['other']).state_dict()
                        },
                        num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path,
                                                            steps)
                    save_checkpoint(
                        checkpoint_path, {
                            'vocals_mpd': (mpds['vocals'].module
                                    if h.num_gpus > 1 else mpds['vocals']).state_dict(),
                            'vocals_msd': (msds['vocals'].module
                                    if h.num_gpus > 1 else msds['vocals']).state_dict(),
                            'vocals_mstftd': (mstftds['vocals'].module
                                       if h.num_gpus > 1 else mstftds['vocals']).state_dict(),
                            'drums_mpd': (mpds['drums'].module
                                    if h.num_gpus > 1 else mpds['drums']).state_dict(),
                            'drums_msd': (msds['drums'].module
                                    if h.num_gpus > 1 else msds['drums']).state_dict(),
                            'drums_mstftd': (mstftds['drums'].module
                                       if h.num_gpus > 1 else mstftds['drums']).state_dict(),
                            'bass_mpd': (mpds['bass'].module
                                    if h.num_gpus > 1 else mpds['bass']).state_dict(),
                            'bass_msd': (msds['bass'].module
                                    if h.num_gpus > 1 else msds['bass']).state_dict(),
                            'bass_mstftd': (mstftds['bass'].module
                                       if h.num_gpus > 1 else mstftds['bass']).state_dict(),
                            'other_mpd': (mpds['other'].module
                                    if h.num_gpus > 1 else mpds['other']).state_dict(),
                            'other_msd': (msds['other'].module
                                    if h.num_gpus > 1 else msds['other']).state_dict(),
                            'other_mstftd': (mstftds['other'].module
                                       if h.num_gpus > 1 else mstftds['other']).state_dict(),
                            'mixture_mpd': (mpds['mixture'].module
                                    if h.num_gpus > 1 else mpds['mixture']).state_dict(),
                            'mixture_msd': (msds['mixture'].module
                                    if h.num_gpus > 1 else msds['mixture']).state_dict(),
                            'mixture_mstftd': (mstftds['mixture'].module
                                       if h.num_gpus > 1 else mstftds['mixture']).state_dict(),
                            'vocals_optim_g':
                            optim_gs['vocals'].state_dict(),
                            'vocals_optim_d':
                            optim_ds['vocals'].state_dict(),
                            'drums_optim_g':
                            optim_gs['drums'].state_dict(),
                            'drums_optim_d':
                            optim_ds['drums'].state_dict(),
                            'bass_optim_g':
                            optim_gs['bass'].state_dict(),
                            'bass_optim_d':
                            optim_ds['bass'].state_dict(),
                            'other_optim_g':
                            optim_gs['other'].state_dict(),
                            'other_optim_d':
                            optim_ds['other'].state_dict(),
                            'mixture_optim_d':
                            optim_ds['mixture'].state_dict(),

                            'vocals_optim_g_mixvae':
                            optim_g_mixvae['vocals'].state_dict(),
                            'drums_optim_g_mixvae':
                            optim_g_mixvae['drums'].state_dict(),
                            'bass_optim_g_mixvae':
                            optim_g_mixvae['bass'].state_dict(),
                            'other_optim_g_mixvae':
                            optim_g_mixvae['other'].state_dict(),

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
                        sw.add_scalar("training/mel_error_source", mel_error_source, steps)
                        sw.add_scalar("training/mel_error_mixture", mel_error_mixture, steps)
                    else:
                        sw.add_scalar("training/gen_loss_total_unsup", loss_gen_all, steps)
                        sw.add_scalar("training/mel_error_unsup", mel_error_unsup, steps)


                # Validation
                # if steps % a.validation_interval == 0 and steps != 0:
                if steps % a.validation_interval == 0 or exp_continue:
                # if steps % 10 == 0 or exp_continue:
                    exp_continue = False
                # if steps % 10 == 0 and steps != 0:
                    for source_type in source_types:
                        mixtureVAEs[source_type].eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    val_vocals_err_tot = 0
                    val_drums_err_tot = 0
                    val_bass_err_tot = 0
                    val_other_err_tot = 0

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

                            # inference for each source type
                            for source_type in source_types:
                                batch = mixtureVAEs[source_type](batch, separate=True, sample_posterior=True)
                                batch = mixtureVAEs[source_type](batch, separate=False, sample_posterior=True, decode=True, train_source_decoder=False, train_source_encoder=True)
                            batch['mixture_dec_fma'] = batch['vocals_dec_fma'] + batch['drums_dec_fma'] + batch['bass_dec_fma'] + batch['other_dec_fma']

                            mix_audio = batch['mixture']
                            mixture_fma = batch['mixture_fma']
                            # groundtruth sources
                            if j <= 8:
                                sw.add_audio('mixture/y_hat_{}'.format(j), mix_audio, steps, h.sampling_rate)
                                sw.add_audio('unsupervised_mixture/y_hat_{}'.format(j), mixture_fma, steps, h.sampling_rate)


                            for source_type in source_types:
                                y = batch[source_type]
                                # y_drums = batch['drums']
                                # y_bass = batch['bass']
                                # y_other = batch['other']


                                # mixture separation samples after decoding
                                y_source = batch[source_type+'_dec']
                                y_mix = batch[source_type+'_dec_mix']

                                dec_mixture_fma = batch['mixture_dec_fma']
                                mixture_fma_mel = default_mel(mixture_fma, h)
                                dec_mixture_fma_mel = default_mel(dec_mixture_fma, h)


                                y_mel = default_mel(y, h)
                                y_mix_mel = default_mel(y_mix, h)
                                y_source_mel = default_mel(y_source, h)

                                i_size = min(y_mel.size(2), y_mix_mel.size(2))
                                
                                val_error_current = F.l1_loss(y_mel[:, :, :i_size], y_mix_mel[:, :, :i_size]).item()
                                val_error_original = F.l1_loss(y_mel[:, :, :i_size], y_source_mel[:, :, :i_size]).item()
                                val_error_unsupervised = F.l1_loss(mixture_fma_mel[:, :, :i_size], dec_mixture_fma_mel[:, :, :i_size]).item()

                                if source_type=='vocals':
                                    val_vocals_err_tot += val_error_current
                                elif source_type=='drums':
                                    val_drums_err_tot += val_error_current
                                elif source_type=='bass':
                                    val_bass_err_tot += val_error_current
                                elif source_type=='other':
                                    val_other_err_tot += val_error_current

                                val_err_tot += val_error_current
                                val_err_self_tot += val_error_original
                                val_unsup_err_tot += val_error_unsupervised

                                if j <= 12:
                                    sw.add_audio('gt/'+source_type+'/y_{}'.format(j), y, steps, h.sampling_rate)
                                    sw.add_audio('mix_generated/'+source_type+'/y_hat_{}'.format(j), y_mix, steps, h.sampling_rate)
                                    sw.add_audio('self_generated/'+source_type+'/y_hat_{}'.format(j), y_source, steps, h.sampling_rate)
                                    sw.add_audio('unsupervised/'+source_type+'/y_hat_{}'.format(j), batch[source_type+'_dec_fma'], steps, h.sampling_rate)

                        val_err_vocals = val_vocals_err_tot / (j+1)
                        val_err_drums = val_drums_err_tot / (j+1)
                        val_err_bass = val_bass_err_tot / (j+1)
                        val_err_other = val_other_err_tot / (j+1)

                        val_err = val_err_tot / (j + 1)
                        val_err_ori = val_err_self_tot / (j + 1)
                        val_err_unsup = val_unsup_err_tot / (j + 1)

                        sw.add_scalar("validation/mel_spec_error_vocals", val_err_vocals, steps)
                        sw.add_scalar("validation/mel_spec_error_drums", val_err_drums, steps)
                        sw.add_scalar("validation/mel_spec_error_bass", val_err_bass, steps)
                        sw.add_scalar("validation/mel_spec_error_other", val_err_other, steps)

                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        sw.add_scalar("validation/mel_spec_error_original", val_err_ori, steps)
                        sw.add_scalar("validation/val_err_unsup", val_err_unsup, steps)

                    for source_type in source_types:
                        mixtureVAEs[source_type].train()

            steps += 1
        
            if steps >= a.supervised_steps and (steps % a.lr_decay_step == 0):
                for source_type in source_types:
                    scheduler_gs[source_type].step()
                    scheduler_ds[source_type].step()
                scheduler_ds['mixture'].step()

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
    parser.add_argument('--lr_decay_step', default=500, type=int)
    parser.add_argument('--update_ratio', default=10000, type=int)
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