import torch
import torch.nn as nn
from models.models_subband2_mixturevae import MixtureEncoder
from models.SourceVAESubband2 import SourceVAE
from models.distributions2d import DiagonalGaussianDistribution
import torchaudio.transforms as T
import torch.nn.functional as F

from models.MixtureVAESingle import MixtureVAE

class MixtureVAEFull(nn.Module):
    def __init__(
        self,
        h,
        load_model=True,
        model_ckpt=None,
    ):
        super().__init__()

        '''
            sourcevae checkpoints: sourcevae_ckpts(disctionary)
        '''
        source_types = ['vocals', 'drums', 'bass', 'other']
        self.source_type = source_types

        mixturevaes = []
        for source_type in source_types:
            h.source_type = source_type
            if load_model:
                mixturevaes.append(MixtureVAE(h, load_model=True, model_ckpt=model_ckpt[source_type], load_source_vae=False, sourcevae_ckpt=None))
                print('mixture vae load finished')
            else:
                mixturevaes.append(MixtureVAE(h, load_model=False, model_ckpt=None, load_source_vae=False, sourcevae_ckpt=None))
        self.mixturevaes = nn.ModuleList(mixturevaes)
        # for mixturevae in self.mixturevaes:
        #     print(mixturevae.source_type, mixturevae.sourcevae.source_type)

    def encode(self, x):
        pos_vocals = self.mixturevaes[0].encode(x)
        pos_drums = self.mixturevaes[1].encode(x)
        pos_bass = self.mixturevaes[2].encode(x)
        pos_other = self.mixturevaes[3].encode(x)

        # pos_vocals, pos_drums, pos_bass, pos_other = DiagonalGaussianDistribution(moments_vocals) \
        #                                             , DiagonalGaussianDistribution(moments_drums) \
        #                                             , DiagonalGaussianDistribution(moments_bass) \
        #                                             , DiagonalGaussianDistribution(moments_other)
        return pos_vocals, pos_drums, pos_bass, pos_other

    def decode(self, z_vocals, z_drums, z_bass, z_other):
        dec_vocals, dec_drums, dec_bass, dec_other = self.mixturevaes[0].decode(z_vocals) \
                                                    , self.mixturevaes[1].decode(z_drums) \
                                                    , self.mixturevaes[2].decode(z_bass) \
                                                    , self.mixturevaes[3].decode(z_other)
        return dec_vocals, dec_drums, dec_bass, dec_other
                                                    

    def forward(self, batch, sample_posterior=True, unsupervised=True, output_source=True, supervised=True):
        '''
            the forward function should do two parts:
                1. for unsupervised mixture, generate all the sources, and calculate kl loss for all the latents, and output one estimated mixture
                2. for supervised mixture, do as before for all the models, add one more estimated mixture
        '''

        # unsupervised inference
        if unsupervised:
            input_fma = batch['mixture_fma'] #.clone() # bs, 1, T
            pos_vocals, pos_drums, pos_bass, pos_other = self.encode(input_fma)
            if sample_posterior:
                z_vocals_fma, z_drums_fma, z_bass_fma, z_other_fma = pos_vocals.sample(), pos_drums.sample(), pos_bass.sample(), pos_other.sample()
            else:
                z_vocals_fma, z_drums_fma, z_bass_fma, z_other_fma = pos_vocals.mode(), pos_drums.mode(), pos_bass.mode(), pos_other.mode()
            batch['unsupervised_kl'] = pos_vocals.kl() + pos_drums.kl() + pos_bass.kl() + pos_other.kl()
            dec_vocals_fma, dec_drums_fma, dec_bass_fma, dec_other_fma = self.decode(z_vocals_fma, z_drums_fma, z_bass_fma, z_other_fma)
            dec_mixture_fma = dec_vocals_fma + dec_drums_fma + dec_bass_fma + dec_other_fma
            batch['dec_mixture_fma'] = dec_mixture_fma
            if output_source:
                batch['dec_vocals_fma'] = dec_vocals_fma
                batch['dec_drums_fma'] = dec_drums_fma
                batch['dec_bass_fma'] = dec_bass_fma
                batch['dec_other_fma'] = dec_other_fma

        if supervised:
            if 'mixture' in batch.keys():
                batch = self.mixturevaes[0](batch, sample_posterior=sample_posterior, decode=True, train_source_decoder=True, train_source_encoder=True)
                batch = self.mixturevaes[1](batch, sample_posterior=sample_posterior, decode=True, train_source_decoder=True, train_source_encoder=True)
                batch = self.mixturevaes[2](batch, sample_posterior=sample_posterior, decode=True, train_source_decoder=True, train_source_encoder=True)
                batch = self.mixturevaes[3](batch, sample_posterior=sample_posterior, decode=True, train_source_decoder=True, train_source_encoder=True)
            batch['mixture_dec_mix'] = batch['vocals_dec_mix'] + batch['drums_dec_mix'] + batch['bass_dec_mix'] + batch['other_dec_mix']
        return batch

# Helper function to select parameters
def get_parameters(model, keyword):
    for name, param in model.named_parameters():
        if keyword in name:
            yield param

if __name__=='__main__':
    from models.models_hificodec import *
    from models.env import *
    import json
    from data.dataset_musdb import dataset_musdb, collate_func_musdb
    from data.dataset_vocal import dataset_vocal, collate_func_vocals
    from data.dataset_semi_supervised import dataset_semi_supervised, collate_func_semi_supsevised
    from torch.utils.data import DistributedSampler, DataLoader
    import os

    with open('../configs/config_MixtureVAE_fma_musdb.json') as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    musdb_root = '/data/romit/alan/musdb18'
    dataset = dataset_semi_supervised(
        musdb_root='/data/romit/alan/musdb18',
        fma_root='/data/romit/alan/fma/fma',
        sample_rate=16000,
        mode='train',
        seconds=1,
        p=1,
    )

    collate_func = collate_func_semi_supsevised
    batch = collate_func([dataset[0]])

    for key in batch.keys():
        print(key, batch[key].shape)
        batch[key] = batch[key].cuda(0)

    ckpt_dir = h.mixvae_ckpt_dir
    ckpt_vocals = torch.load(os.path.join(ckpt_dir, 'ckpt_vocals'))
    ckpt_drums = torch.load(os.path.join(ckpt_dir, 'ckpt_drums'))
    ckpt_bass = torch.load(os.path.join(ckpt_dir, 'ckpt_bass'))
    ckpt_other = torch.load(os.path.join(ckpt_dir, 'ckpt_other'))
    model_ckpts = {
        'vocals':ckpt_vocals, 'drums':ckpt_drums, 'bass':ckpt_bass, 'other':ckpt_other, 
    }

    mixturevae_full = MixtureVAEFull(
        h,
        load_model=True,
        model_ckpt=model_ckpts).cuda(0)
    
    batch = mixturevae_full(batch, unsupervised=False)
    
    for key in batch.keys():
        print(key, batch[key].shape)
