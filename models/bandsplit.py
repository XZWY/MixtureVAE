import torch
import torch.nn as nn
import typing as tp
import torch
import typing as tp


def get_fftfreq(
        sr: int = 44100,
        n_fft: int = 2048
) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    """
    out = sr * torch.fft.fftfreq(n_fft)[:n_fft // 2 + 1]
    out[-1] = sr // 2
    return out


def get_subband_indices(
        freqs: torch.Tensor,
        splits: tp.List[tp.Tuple[int, int]],
) -> tp.List[tp.Tuple[int, int]]:
    """
    Computes subband frequency indices with given bandsplits
    """
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices


def freq2bands(
        bandsplits: tp.List[tp.Tuple[int, int]],
        sr: int = 44100,
        n_fft: int = 2048
) -> tp.List[tp.Tuple[int, int]]:
    """
    Returns start and end FFT indices of given bandsplits
    """
    freqs = get_fftfreq(sr=sr, n_fft=n_fft)
    band_indices = get_subband_indices(freqs, bandsplits)
    return band_indices

class BandSplitModule(nn.Module):
    """
    BandSplit (1st) Module of BandSplitRNN.
    Separates input in k subbands and runs through LayerNorm+FC layers.
    """

    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            t_timesteps: int = 517,
            fc_dim: int = 128,
            complex_as_channel: bool = True,
            is_mono: bool = False,
            is_layernorm=False
    ):
        super(BandSplitModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.is_layernorm = is_layernorm
        if is_layernorm:
            self.layernorms = nn.ModuleList([
                nn.LayerNorm([(e - s) * frequency_mul, t_timesteps])
                for s, e in self.bandwidth_indices
            ])
        self.fcs = nn.ModuleList([
            nn.Linear((e - s) * frequency_mul, fc_dim)
            for s, e in self.bandwidth_indices
        ])

    def generate_subband(
            self,
            x: torch.Tensor
    ) -> tp.Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, n_channels, freq, time]
        Output: [batch_size, k_subbands, time, fc_output_shape]
        """
        xs = []
        for i, x in enumerate(self.generate_subband(x)):
            B, C, F, T = x.shape
            # view complex as channels
            if x.dtype == torch.cfloat:
                x = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
            # from channels to frequency
            x = x.reshape(B, -1, T)
            # run through model
            if self.is_layernorm:
                x = self.layernorms[i](x)
            x = x.transpose(-1, -2)
            x = self.fcs[i](x)
            # print(x.shape)
            xs.append(x)
        return torch.stack(xs, dim=3).permute(0,2,1,3).contiguous()

class BandMergeModule(nn.Module):
    """
    BandMerge (1st) Module of BandSplitRNN.
    Separates input in k subbands and runs through LayerNorm+FC layers.
    """

    def __init__(
            self,
            bandsplits: tp.List[tp.Tuple[int, int]],
            fc_dim: int = 128,
            sr=16000,
            n_fft=1024
    ):
        super(BandMergeModule, self).__init__()
        self.bandsplits = bandsplits
        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.fcs = nn.ModuleList([
            nn.Linear(fc_dim, (e - s) * 2)
            for s, e in self.bandwidth_indices
        ])
        
    def forward(self, bands: torch.Tensor):
        # bands bs, hidden_dim, T, num_bands
        bands = bands.permute(0,2,1,3) # bs, T, hidden_dim, num_bands
        B, T, H, N = bands.shape
        out = []
        for band_idx in range(bands.shape[-1]):
            band_output = self.fcs[band_idx](bands[..., band_idx]).view(B, T, 2, -1)
            out.append(band_output)
        out = torch.cat(out, dim=-1).permute(0,3,1,2)
        
        return out
    
# if __name__ == '__main__':
#     freqs_splits = [
#         (1000, 100),
#         (4000, 250),
#         (7500, 500),
#     ]
#     sr = 16000
#     n_fft = 1024

#     out = freq2bands(freqs_splits, sr, n_fft)

#     sum_tuples = 0
#     for tup in out:
#         # print(tup)
#         sum_tuples += (tup[1] - tup[0])
#     # print(sum_tuples)

#     assert sum_tuples == n_fft // 2 + 1

#     print(f"Input:\n{freqs_splits}\n{sr}\n{n_fft}\nOutput:{out}")

if __name__ == '__main__':
    batch_size, n_channels, freq, time = 2, 2, 513, 100

    in_features = torch.rand(batch_size, n_channels, freq, time, dtype=torch.float32)

    cfg = {
        "sr": 16000,
        "complex_as_channel": False,
        "is_mono": False,
        "n_fft": 1024,
        "bandsplits": [
            (1000, 100),
            (4000, 250),
            (7500, 500),
        ],
        "t_timesteps": 100,
        "fc_dim": 128
    }

    bandsplit = BandSplitModule(**cfg)
    bandmerge = BandMergeModule(
        bandsplits=[
            (1000, 100),
            (4000, 250),
            (7500, 500),],
        fc_dim=128,
        sr=16000,
        n_fft=1024
        )

    in_features = torch.rand(batch_size, n_channels, freq, time, dtype=torch.float32)
    print(in_features.shape)
    out_features = bandsplit(in_features)
    print(out_features.shape)
    stft = bandmerge(out_features)
    print(stft.shape)

    # print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")