import torch
import torch.nn as nn
from models.models_subband2_mixturevae import MixtureEncoder
from models.SourceVAESubband2 import SourceVAE
from models.distributions2d import DiagonalGaussianDistribution
import torchaudio.transforms as T
import torch.nn.functional as F

class MixtureVAE(nn.Module):
    def __init__(
        self,
        h,
        load_model=True,
        model_ckpt=None,
        load_source_vae=True,
        sourcevae_ckpt=None
    ):
        super().__init__()

        '''
            sourcevae checkpoints: sourcevae_ckpts(disctionary)
        '''
        self.source_type = h.source_type
        self.MixEncoder = MixtureEncoder(
            channels=[2, 64, 64, 64, 64, 128, 128],
            bottleneck_dim=h.bottleneck_dimension
        )

        self.embed_dim = h.bottleneck_dimension
        self.flag_first_run=True
        
        self.sourcevae = SourceVAE(h)
        if load_model:
            assert model_ckpt!=None, 'model ckpt not provided!!!!!!!!!!!!!'
            self.load_state_dict(model_ckpt['mixturevae'])
            print('--------------------------finish loading mixturevae checkpoints------------------------------------')

        elif load_source_vae:
            assert sourcevae_ckpt!=None, 'source vae checkpoints not provided!!!!!'
            self.sourcevae.load_state_dict(sourcevae_ckpt['sourcevae'])
            print('--------------------------finish loading sourcevae checkpoints------------------------------------')
        
        self.freeze_source_vae = False

    def encode(self, x):
        '''
        x: bs, 1, T
        '''
        moments = self.MixEncoder(x) # bs*4, n_emb * 2, T, F
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        source_dec = self.sourcevae.decode(z)
        return source_dec

    def separate(self, batch, sample_posterior=True):
        posterior = self.encode(batch['mixture_fma'])
        if sample_posterior:
            z_out = posterior.sample()
        else:
            z_out = posterior.mode()
        batch[self.source_type+'_unsupervised_kl'] = posterior.kl()
        batch[self.source_type+'_dec_fma'] = self.decode(z_out)
        return batch

    def freeze_source_vae(self):
        for param in self.sourcevae.parameters():
            param.require_grad = False

    def forward(self, batch, separate=False, sample_posterior=True, decode=False, train_source_decoder=False, train_source_encoder=False, freeze_source_vae=False):
        '''
            forward mostly for training:
                decode: whether we want to decode source signals, decode would also create a decoded mixture_hat, which is a sum of decoded sources
                train_source_encoder: the source encoder's forward's gradients are calculated, weights updatable
                train_source_decoder: the source decoder's forward's gradients are calculated, weight updatable
            
            a few notices for future reference:
                supervised training:
                    1. posterior matching only: decode=False, train_source_decoder=False, train_source_encoder=False
                    2. posterior matching with learnable SourceVAE and source GAN: decode=True, train_source_encoder=True, train_source_decoder=True
                supervised evaluation:
                    decode = True, train_encoder=False, train_decoder=False
                unsupervised training:
                    1. no source encoder needed even, need to modify this function or create a new one, decode=True, train_source_decoder=True


        '''

        if separate:
            posterior = self.encode(batch['mixture_fma'])
            if sample_posterior:
                z_out = posterior.sample()
            else:
                z_out = posterior.mode()
            batch[self.source_type+'_unsupervised_kl'] = posterior.kl()
            batch[self.source_type+'_dec_fma'] = self.decode(z_out)
            return batch

        input = batch['mixture']#.clone() # bs, 1, T

        posterior_mix = self.encode(input)

        if decode:
            # only sample and decode when needed
            if sample_posterior:
                z_mix = posterior_mix.sample()
            else:
                z_mix = posterior_mix.mode()

            if self.flag_first_run:
                print("Latent size: ", z_mix.size())
                self.flag_first_run = False

            if not train_source_decoder:
                with torch.no_grad():
                    dec_mix = self.decode(z_mix)
            else:
                dec_mix = self.decode(z_mix)

            batch[self.source_type+'_dec_mix'] = dec_mix # vocals_dec_mix

        # need to encode groudtruth source latents posteriors Zk | Sk
        if train_source_encoder:
            posterior_source = self.sourcevae.encode(batch[self.source_type])
            batch[self.source_type+'_loss_KLD'] = posterior_source.kl()

            if sample_posterior:
                z_source = posterior_source.sample()
            else:
                z_source = posterior_source.mode()

            dec_source = self.sourcevae.decode(z_source)

            batch[self.source_type+'_dec'] = dec_source # vocals_dec

        else:
            with torch.no_grad():
                posterior_source = self.sourcevae.encode(batch[self.source_type])

        # calculate posterior matching loss KL divergence between Zk|X and Zk|Sk
        batch[self.source_type+'_loss_posterior_matching'] = posterior_mix.kl(posterior_source)

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
    from torch.utils.data import DistributedSampler, DataLoader
    import os

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    with open('../config_MixtureVAE_other.json') as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    musdb_root = '/data/romit/alan/musdb18'
    dataset = dataset_musdb(
        root_dir=musdb_root,
        sample_rate=16000,
        mode='train',
        source_types=['other'],
        mixture=True,
        seconds=1,
        len_ds=5000)

    collate_func = collate_func_musdb
    batch = collate_func([dataset[0]])

    for key in batch.keys():
        print(key, batch[key].shape)
        batch[key] = batch[key].cuda(1)

    ckpt_dir = '/data/romit/alan/MixtureVAE/ckpt_source_vae'
    # ckpt_vocals = torch.load(os.path.join(ckpt_dir, 'ckpt_vocals'))
    # ckpt_drums = torch.load(os.path.join(ckpt_dir, 'ckpt_drums'))
    # ckpt_bass = torch.load(os.path.join(ckpt_dir, 'ckpt_bass'))
    ckpt_other = torch.load(os.path.join(ckpt_dir, 'ckpt_other'))
    # sourcevae_ckpts = {
    #     'vocals':ckpt_vocals, 'drums':ckpt_drums, 'bass':ckpt_bass, 'other':ckpt_other
    # }
    model_ckpt_dir = '/data/romit/alan/MixtureVAE/log_files/log_mixvae_other/ckpt_pretrained_mixvae_50000'
    model_ckpt = torch.load(model_ckpt_dir)
    # sourcevae = SourceVAE(h)
    # sourcevae.source_type = 'other'
    # sourcevae.load_state_dict(sourcevae_ckpts['other']['sourcevae'])
    mixturevae = MixtureVAE(h, load_model=True, model_ckpt=model_ckpt, load_source_vae=True, sourcevae_ckpt=ckpt_other).cuda(1)
    # for n, p in mixturevae.named_parameters():
    #     print(n, p.shape, 'MixEncoder' in n)
    print('--------------------------------')
    # g_parameters = get_parameters(mixturevae, 'MixEncoder')
    # for p in g_parameters:
    #     print(p.shape)

    # device = "cuda:6"
    # ckpt = torch.load('/data/romit/alan/MixtureVAE/log_files/log_mixvae_trial/g_00000000', map_location=device)
    # print(ckpt.keys())
    # mixturevae.load_state_dict(ckpt['mixturevae'])
    mixturevae(batch, decode=True, train_source_decoder=True, train_source_encoder=True)

    for key in batch.keys():
        print(key, batch[key].shape)