import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio.transforms as T

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

if __name__ == '__main__':
    s = torch.randn(2, 32000).cuda()
    s_hat = torch.randn(2, 32000).cuda()
    loss = loss_l1(s, s_hat)
    print(loss)

# vae reconstruction loss (frequency dependent mse on compressed spec)
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


