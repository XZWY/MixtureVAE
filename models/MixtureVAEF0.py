import torch
import torch.nn as nn
from models.models_F0_separator import MixtureEncoder
from models.SourceVAEAllClass import SourceVAE
from models.distributions2d import DiagonalGaussianDistribution
import torchaudio.transforms as T
import torch.nn.functional as F

def snr(y, y_hat, epsilon=1e-8):
    '''
    s: bs, T
    s_hat: bs, T
    '''
    return 10*torch.log10((y**2).sum(1)+epsilon) - 10*torch.log10(((y - y_hat)**2).sum(1)+epsilon)

class MixtureVAE(nn.Module):
    def __init__(
        self,
        h,
        load_model=False,
        model_ckpt=None,
        load_source_vae=False,
        sourcevae_ckpt=None
    ):
        super().__init__()

        '''
            sourcevae checkpoints: sourcevae_ckpts(disctionary)
        '''
        # self.source_type = h.source_type
        self.MixEncoder = MixtureEncoder(
            # channels=[2, 64, 128, 256, 256, 256, 256],
            channels=[2, 64, 64, 128, 128, 256, 256],
            bottleneck_dim=h.bottleneck_dimension
        )

        # assert sourcevae_ckpt!=None, 'source vae checkpoints not provided!!!!!'

        classes = ['vocals', 'drums', 'bass', 'other']
        self.classes = classes
        self.sourcevae = SourceVAE(h)


        if load_model:
            assert model_ckpt!=None, 'model ckpt not provided!!!!!!!!!!!!!'
            self.load_state_dict(model_ckpt['mixturevae'])
            print('--------------------------finish loading mixturevae checkpoints------------------------------------')

        elif load_source_vae:
            assert sourcevae_ckpt!=None, 'source vae checkpoints not provided!!!!!'
            self.sourcevae.load_state_dict(sourcevae_ckpt['sourcevae'])
            print('--------------------------finish loading sourcevae checkpoints------------------------------------')


        self.embed_dim = h.bottleneck_dimension
        
        # self.mean, self.std = None, None
        
        self.flag_first_run=True
        
        # self.reconstruction_loss = loss_vae_reconstruction(h)

    def encode(self, x):
        '''
        x: bs, 1, T
        '''
        bs = x.shape[0]

        moments = self.MixEncoder(x) # bs, 4, n_emb * 2, T, F
        bs, _, H, T, F = moments.shape
        moments = moments.permute(1,0,2,3,4).contiguous().view(4*bs, H, T, F)
        _, H, T, F = moments.shape
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        dec = self.sourcevae.decode(z)
        return dec

    def forward(self, batch, sample_posterior=True, decode=False, train_source_decoder=False, train_source_encoder=False):
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
        input = batch['mixture']#.clone() # bs, 1, T

        posterior_mix = self.encode(input) # encode scaled input (0.05, 0.95)

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
            batch['output_mix'] = dec_mix
            ref = torch.cat([batch['vocals'], batch['drums'], batch['bass'], batch['other']], dim=0)
            batch['reference'] = ref
            batch['loss_snr_mixture'] = -snr(ref.squeeze(1), dec_mix.squeeze(1))

            B, _, T = dec_mix.shape
            dec_mix = dec_mix.view(4, B//4, 1, T)

            batch['mixture_hat'] = dec_mix.sum(0)

        # need to encode groudtruth source latents posteriors Zk | Sk
        if train_source_encoder:

            sources = torch.cat([batch['vocals'], batch['drums'], batch['bass'], batch['other']], dim=0)#.clone() # bs, 1, T
            ref = sources.clone()
            batch['reference'] = ref
            posterior_source = self.sourcevae.encode(sources)

            batch['loss_KLD'] = posterior_source.kl()

            if sample_posterior:
                z_source = posterior_source.sample()
            else:
                z_source = posterior_source.mode()

            dec_source = self.sourcevae.decode(z_source)

            batch['output_source'] = dec_source

            batch['loss_snr_source'] = -snr(ref.squeeze(1), dec_source.squeeze(1))
        else:
            with torch.no_grad():
                sources = torch.cat([batch['vocals'], batch['drums'], batch['bass'], batch['other']], dim=0)#.clone() # bs, 1, T
                ref = sources.clone()
                batch['reference'] = ref
                posterior_source = self.sourcevae.encode(sources)

        # calculate posterior matching loss KL divergence between Zk|X and Zk|Sk
        batch['loss_posterior_matching'] = posterior_mix.kl(posterior_source)

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
    from models.SourceVAESB import SourceVAESB
    from models.MixtureVAEF0 import MixtureVAE
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

    with open('../config_MixtureVAE.json') as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    musdb_root = '/data/romit/alan/musdb18'
    dataset = dataset_musdb(
        root_dir=musdb_root,
        sample_rate=16000,
        mode='train',
        source_types=['vocals', 'drums', 'bass', 'other'],
        mixture=True,
        seconds=0.5,
        len_ds=5000)

    collate_func = collate_func_musdb
    batch = collate_func([dataset[0], dataset[1]])

    for key in batch.keys():
        print(key, batch[key].shape)
        batch[key] = batch[key].cuda(1)

    ckpt_dir = '/data/romit/alan/MixtureVAE/ckpt_source_vae'
    ckpt_vocals = torch.load(os.path.join(ckpt_dir, 'g_00070000'))
    # ckpt_drums = torch.load(os.path.join(ckpt_dir, 'ckpt_drums'))
    # ckpt_bass = torch.load(os.path.join(ckpt_dir, 'ckpt_bass'))
    # ckpt_other = torch.load(os.path.join(ckpt_dir, 'ckpt_other'))
    # sourcevae_ckpts = {
    #     'vocals':ckpt_vocals, 'drums':ckpt_drums, 'bass':ckpt_bass, 'other':ckpt_other
    # }

    # sourcevae = SourceVAE(h)
    # sourcevae.source_type = 'other'
    # sourcevae.load_state_dict(sourcevae_ckpts['other']['sourcevae'])

    mixturevae = MixtureVAE(h, load_source_vae=True, sourcevae_ckpt=ckpt_vocals).cuda(1)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the number of parameters
    num_params = count_parameters(mixturevae.MixEncoder)
    print(f'The model has {num_params} trainable parameters.')
    # for n, p in mixturevae.named_parameters():
    #     print(n, p.shape, 'MixEncoder' in n)
    # print('--------------------------------')
    # g_parameters = get_parameters(mixturevae, 'MixEncoder')
    # for p in g_parameters:
    #     print(p.shape)

    # device = "cuda:6"
    # ckpt = torch.load('/data/romit/alan/MixtureVAE/log_files/log_mixvae_trial/g_00000000', map_location=device)
    # print(ckpt.keys())
    # mixturevae.load_state_dict(ckpt['mixturevae'])
    mixturevae(batch, sample_posterior=True, decode=True, train_source_decoder=True, train_source_encoder=True)
    print('--------------------------------------------after running----------------------------------------------')
    for key in batch.keys():
        print(key, batch[key].shape)