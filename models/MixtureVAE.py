import torch
import torch.nn as nn
from models.models_subband2_separator import MixtureEncoder
from SourceVAESubband2 import SourceVAE
from models.distributions2d import DiagonalGaussianDistribution
import torchaudio.transforms as T
import torch.nn.functional as F

class MixtureVAE(nn.Module):
    def __init__(
        self,
        h,
        sourcevae_ckpts=None
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

        # assert sourcevae_ckpts==None, 'source vae checkpoints not provided!!!!!'

        classes = ['vocals', 'drums', 'bass', 'other']
        self.classes = classes
        sourcevaes = []
        for n_class in range(len(classes)):
            vae = SourceVAE(h)
            # SourceVAE.load_state_dict(sourcevae_ckpts[classes[n_class]['sourcevae']])
            SourceVAE.source_type=classes[n_class]
            sourcevaes.append(vae)
        self.sourcevaes = nn.ModuleList(sourcevaes)

        self.embed_dim = h.bottleneck_dimension
        
        # self.mean, self.std = None, None
        
        self.flag_first_run=True
        
        # self.reconstruction_loss = loss_vae_reconstruction(h)

    def encode(self, x):
        '''
        x: bs, 1, T
        '''
        bs = x.shape[0]
        indices = torch.tensor([0,1,2,3]).unsqueeze(0).expand(bs, -1).to(x.device) # bs, 4
        indices = indices.reshape(-1)

        input = x.repeat(1, 4, 1).view(bs*4, 1, -1) # bs*4, 1, T
        moments = self.MixEncoder(input, indices) # bs*4, n_emb * 2, T, F
        _, H, T, F = moments.shape
        moments = moments.view(bs, 4, H, T, F)
        posterior_vocal = DiagonalGaussianDistribution(moments[:, 0, ...])
        posterior_drums = DiagonalGaussianDistribution(moments[:, 1, ...])
        posterior_bass = DiagonalGaussianDistribution(moments[:, 2, ...])
        posterior_other = DiagonalGaussianDistribution(moments[:, 3, ...])
        return (posterior_vocal, posterior_drums, posterior_bass, posterior_other)

    def decode(self, z_vocals, z_drums, z_bass, z_other):
        vocals_dec = self.sourcevaes[0].decode(z_vocals)
        drums_dec = self.sourcevaes[1].decode(z_drums)
        bass_dec = self.sourcevaes[2].decode(z_bass)
        other_dec = self.sourcevaes[3].decode(z_other)
        return (vocals_dec, drums_dec, bass_dec, other_dec)

    def forward(self, batch, sample_posterior=True, decode=False, train_source_decoder=False, train_source_encoder=False):
    
        input = batch['mixture']#.clone() # bs, 1, T

        posterior_vocals_mix, posterior_drums_mix, posterior_bass_mix, posterior_other_mix = self.encode(input) # encode scaled input (0.05, 0.95)

        if decode:
            # only sample and decode when needed
            if sample_posterior:
                z_vocal_mix, z_drums_mix, z_bass_mix, z_other_mix = posterior_vocals_mix.sample(), posterior_drums_mix.sample(), posterior_bass_mix.sample(), posterior_other_mix.sample()
            else:
                z_vocal_mix, z_drums_mix, z_bass_mix, z_other_mix = posterior_vocals_mix.mode(), posterior_drums_mix.mode(), posterior_bass_mix.mode(), posterior_other_mix.mode()

            if self.flag_first_run: 
                print("Latent size: ", z_vocal_mix.size())
                self.flag_first_run = False

            if not train_source_decoder:
                with torch.no_grad():
                    vocals_dec_mix, drums_dec_mix, bass_dec_mix, other_dec_mix = self.decode(z_vocal_mix, z_drums_mix, z_bass_mix, z_other_mix)
            else:
                vocals_dec_mix, drums_dec_mix, bass_dec_mix, other_dec_mix = self.decode(z_vocal_mix, z_drums_mix, z_bass_mix, z_other_mix)

            batch['vocals_dec_mix'] = vocals_dec_mix
            batch['drums_dec_mix'] = drums_dec_mix
            batch['bass_dec_mix'] = bass_dec_mix
            batch['other_dec_mix'] = other_dec_mix
            batch['mixture_hat'] = vocals_dec_mix + drums_dec_mix + bass_dec_mix + other_dec_mix

        # need to encode groudtruth source latents posteriors Zk | Sk
        if train_source_encoder:
            posterior_vocals_source = self.sourcevaes[0].encode(batch['vocals'])
            posterior_drums_source = self.sourcevaes[1].encode(batch['drums'])
            posterior_bass_source = self.sourcevaes[2].encode(batch['bass'])
            posterior_other_source = self.sourcevaes[3].encode(batch['other'])
            batch['loss_KLD'] = posterior_vocals_source.kl() + posterior_drums_source.kl() + posterior_bass_source.kl() + posterior_other_source.kl()

            if sample_posterior:
                z_vocal, z_drums, z_bass, z_other = posterior_vocals_source.sample(), posterior_drums_source.sample(), posterior_bass_source.sample(), posterior_other_source.sample()
            else:
                z_vocal, z_drums, z_bass, z_other = posterior_vocals_source.mode(), posterior_drums_source.mode(), posterior_bass_source.mode(), posterior_other_source.mode()

            vocals_dec_source, drums_dec_source, bass_dec_source, other_dec_source = self.sourcevaes[0].decode(z_vocal), self.sourcevaes[1].decode(z_drums), self.sourcevaes[2].decode(z_bass), self.sourcevaes[3].decode(z_other)

            batch['vocals_dec'] = vocals_dec_source
            batch['drums_dec'] = drums_dec_source
            batch['bass_dec'] = bass_dec_source
            batch['other_dec'] = other_dec_source

        else:
            with torch.no_grad():
                posterior_vocals_source = self.sourcevaes[0].encode(batch['vocals'])
                posterior_drums_source = self.sourcevaes[1].encode(batch['drums'])
                posterior_bass_source = self.sourcevaes[2].encode(batch['bass'])
                posterior_other_source = self.sourcevaes[3].encode(batch['other'])

        # calculate posterior matching loss KL divergence between Zk|X and Zk|Sk
        batch['loss_posterior_matching'] = posterior_vocals_mix.kl(posterior_vocals_source)\
                                + posterior_drums_mix.kl(posterior_drums_source)\
                                + posterior_bass_mix.kl(posterior_bass_source)\
                                + posterior_other_mix.kl(posterior_other_source)

        return batch

if __name__=='__main__':
    from models.models_hificodec import *
    from models.env import *
    import json
    from data.dataset_musdb import dataset_musdb, collate_func_musdb
    from data.dataset_vocal import dataset_vocal, collate_func_vocals
    from models.SourceVAESB import SourceVAESB
    from models.MixtureVAE import MixtureVAE
    from torch.utils.data import DistributedSampler, DataLoader

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
        seconds=1,
        len_ds=5000)

    collate_func = collate_func_musdb
    batch = collate_func([dataset[0], dataset[1]])

    for key in batch.keys():
        print(key, batch[key].shape)
        batch[key] = batch[key].cuda(6)


    mixturevae = MixtureVAE(h).cuda(6)
    mixturevae(batch, decode=True, train_source_decoder=False, train_source_encoder=True)

    for key in batch.keys():
        print(key, batch[key].shape)