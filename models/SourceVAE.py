import torch
import torch.nn as nn
from models.models_hificodec import Encoder, Generator
from models.distributions import DiagonalGaussianDistribution
import torchaudio.transforms as T
import torch.nn.functional as F

class loss_vae_reconstruction(nn.Module):
    def __init__(
        self,
        h
    ):
        super().__init__()
        self.nfreq = int(h.reconstruction_loss_nfft//2)+1
        self.c = h.reconstruction_loss_c
        
        self.STFT = T.Spectrogram(
            n_fft=h.reconstruction_loss_nfft,
            hop_length=h.reconstruction_hoplength,
            window_fn =torch.hann_window,
            power=None
            )
        
    def drc_stft(self, Y):
        Y_mag = torch.sqrt(Y[..., 0]**2 + Y[..., 1]**2).unsqueeze(-1)
        Y_mag_c = Y_mag**(self.c)
        return Y / (Y_mag + 1e-9) * Y_mag_c
        
    def forward(self, y, y_hat):
        '''
            y: (..., T)
            y_hat: (..., T)
            out: (...,)
        '''
        bs = y.shape[0]
        Y_stft = self.STFT(y) # ..., nfreq, time, 2
        Y_hat_stft = self.STFT(y_hat) # ..., nfreq, time, 2
        
        Y_stft_c, Y_hat_stft_c = self.drc_stft(Y_stft), self.drc_stft(Y_hat_stft)
        loss = ((Y_stft_c - Y_hat_stft_c)**2).sum(-1) # ..., n_freq, time
        
        return torch.mean(loss, dim=[-1, -2]).squeeze(-1)

def snr(y, y_hat, epsilon=1e-8):
    '''
    s: bs, T
    s_hat: bs, T
    '''
    return 10*torch.log10((y**2).sum(1)+epsilon) - 10*torch.log10(((y - y_hat)**2).sum(1)+epsilon)

def loss_l1(s, s_hat):
    '''
    s: bs, T
    s_hat: bs, T
    '''
    l1_time = F.l1_loss(s_hat, s)
    window = (torch.hann_window(512) ** 0.5).to(s.device)
    s_stft = torch.stft(input=s, n_fft=512, hop_length=256, win_length=512, window=window, center=True, pad_mode='reflect', normalized=True, onesided=None, return_complex=True)
    s_hat_stft = torch.stft(input=s_hat, n_fft=512, hop_length=256, win_length=512, window=window, center=True, pad_mode='reflect', normalized=True, onesided=None, return_complex=True)
    s_stft_mag = s_stft.abs()
    s_hat_stft_mag = s_hat_stft.abs()
    
    l1_freq = F.l1_loss(s_hat_stft_mag, s_stft_mag)
    
    return l1_time + l1_freq

class SourceVAE(nn.Module):
    def __init__(
        self,
        h
    ):
        super().__init__()

        self.source_type = h.source_type
        self.encoder = Encoder(h)
        self.decoder = Generator(h)

        self.embed_dim = h.bottleneck_dimension
        
        # self.mean, self.std = None, None
        
        self.flag_first_run=True
        
        # self.reconstruction_loss = loss_vae_reconstruction(h)

    def encode(self, x):
        moments = self.encoder(x) # bs, 
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, batch, sample_posterior=True):
    
        input = batch[self.source_type]#.clone() # bs, 1, T
        ref = input.clone()
        # device = input.device

        # bs = input.shape[0]
        # scale = torch.clamp((0.3*torch.randn(bs, 1, 1) + 0.3), min=0.05, max=0.95).to(device)
        # batch['scale'] = scale
        # input = input / (input.abs().max(dim=2, keepdim=True).values + 1e-8)
        # batch[self.source_type+'_reference'] = input.clone() # rescaled input, -1, 1, used for loss calculation
        posterior = self.encode(input) # encode scaled input (0.05, 0.95)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if self.flag_first_run: 
            print("Latent size: ", z.size())
            self.flag_first_run = False

        dec = self.decode(z)
        
        # add loss here
        batch['output'] = dec
        batch['loss_KLD'] = posterior.kl()
        # batch['loss_reconstruction'] = self.reconstruction_loss(input, dec)
        # batch['loss_l1'] = loss_l1(ref.squeeze(1), dec.squeeze(1))
        # batch['loss_snr'] = -snr(ref.squeeze(1), dec.squeeze(1))

        return batch

    # def forward(self, batch, sample_posterior=False):
        
    #     input = batch[self.source_type].clone() # bs, 1, T
    #     device = input.device

    #     bs = input.shape[0]
    #     scale = torch.clamp((0.3*torch.randn(bs, 1, 1) + 0.3), min=0.05, max=0.95).to(device)
    #     batch['scale'] = scale
    #     input = input / (input.abs().max(dim=2, keepdim=True).values + 1e-8)
    #     batch[self.source_type+'_reference'] = input.clone() # rescaled input, -1, 1, used for loss calculation
    #     input = input * scale
        
    #     import soundfile as sf
        
    #     sf.write('input_to_network.wav', input[0, 0], 16000)
    #     posterior = self.encode(input) # encode scaled input (0.05, 0.95)
    #     if sample_posterior:
    #         z = posterior.sample()
    #     else:
    #         z = posterior.mode()

    #     if self.flag_first_run: 
    #         print("Latent size: ", z.size())
    #         self.flag_first_run = False

    #     dec = self.decode(z)
        
    #     # label is capped to -1, 1, dec is capped with the same division
    #     dec = dec / batch['scale']
        
    #     # add loss here
    #     batch['output'] = dec
    #     batch['loss_KLD'] = posterior.kl()
    #     batch['loss_reconstruction'] = self.reconstruction_loss(batch[self.source_type+'_reference'], dec)
    #     batch['loss_l1'] = loss_l1(batch[self.source_type+'_reference'].squeeze(1), dec.squeeze(1))

    #     return batch

    # def forward(self, input, sample_posterior=True):
    #     posterior = self.encode(input)
    #     if sample_posterior:
    #         z = posterior.sample()
    #     else:
    #         z = posterior.mode()

    #     if self.flag_first_run:
    #         print("Latent size: ", z.size())
    #         self.flag_first_run = False

    #     dec = self.decode(z)

    #     return dec, posterior
    
# class SourceVAEwithVAELoss(nn.Module):
#     def __init__(
#         self,
#         h
#     ):
#         super().__init__()
#         self.sourcevae = SourceVAE(h)
    
#     def forward()