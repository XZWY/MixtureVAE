import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd import Variable

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

LRELU_SLOPE = 0.1

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 3), 1, dilation=(dilation[0], 1),
                               padding=(get_padding(kernel_size, dilation[0]), 1))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(dilation[1], 1),
                               padding=(get_padding(kernel_size, dilation[1]), 2))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(dilation[2], 1),
                               padding=(get_padding(kernel_size, dilation[2]), 2))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 3), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 1))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 2))),
            weight_norm(nn.Conv2d(channels, channels, (kernel_size, 5), 1, dilation=(1, 1),
                               padding=(get_padding(kernel_size, 1), 2))),
        ])
        self.convs2.apply(init_weights)
        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class RNNLayer(torch.nn.Module):
    def __init__(self, channels, hidden_dim):
        super(RNNLayer, self).__init__()
        self.sb_lstm = nn.LSTM(input_size=channels, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fb_lstm = nn.LSTM(input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.linear2 = nn.Linear(channels*2, channels)

    def forward(self, input):
        # input: bs, C, T, F
        bs, C, T, F = input.shape
        out = input

        out = out.permute(0,3,2,1).contiguous().view(bs*F, T, C) # bs*F, T, C
        out, _ = self.sb_lstm(out) # bs*F, T, 2C
        out = out.view(bs, F, T, -1).permute(0,2,1,3).contiguous().view(bs*T, F, -1) # bs*T F, C
        out, _ = self.fb_lstm(out) # bs*T, F, 2c
        out = self.linear2(out).view(bs, T, F, C).permute(0,3,1,2).contiguous() # bs, c, T, F
        return out


class MixtureEncoder(torch.nn.Module):
    def __init__(self, channels=[2, 64, 64, 64, 64, 128, 128], bottleneck_dim=32):
        super(MixtureEncoder, self).__init__()
        self.channels = channels
        f_kernel_size = [5,3,3,3,3,4]
        f_stride_size = [2,2,2,1,1,1]
        resblock_kernel_sizes = [3,7]
        resblock_dilation_sizes = [[1,3,5], [1,3,5]]
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_layers = len(channels) - 1
        self.normalize = nn.ModuleList()

        conv_list = []
        norm_list = []
        res_list = []
        rnn_list = []
        
        self.num_kernels = len(resblock_kernel_sizes)
        for c_idx in range(self.num_layers):
            rnn_list.append(RNNLayer(channels[c_idx+1], channels[c_idx+1]))
            conv_list.append(
                nn.Conv2d(channels[c_idx], channels[c_idx+1], (3, f_kernel_size[c_idx]), stride=(1, f_stride_size[c_idx]), padding=(1,0)),
            )
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes))
                )
            ):
                res_list.append(ResBlock(channels[c_idx+1], k, d))    
                norm_list.append(nn.GroupNorm(1, channels[c_idx+1], eps=1e-6, affine=True))
       
        self.rnn_list = nn.ModuleList(rnn_list)
        self.conv_list = nn.ModuleList(conv_list)
        self.norm_list = nn.ModuleList(norm_list)
        self.res_list = nn.ModuleList(res_list)
        
        self.conv_list.apply(init_weights)
                
        self.conv_post = weight_norm(nn.Conv2d(channels[-1], bottleneck_dim*2, 1, 1, padding=0))
        # self.conv_post.apply(init_weights)
        
        self.window = torch.hann_window(640)


    def forward(self, audio):
        '''
        x: bs, 2, T, F
        out: bs, 256, n_frames, 2
        '''
        audio = audio.squeeze(1)
        device = audio.device
        x = torch.stft(audio, n_fft=640, hop_length=160, win_length=640, window=self.window.to(device), center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        # x: bs, F, T
        x = torch.view_as_real(x).permute(0, 3, 2, 1)
        bs, _, n_frames, n_freqs = x.shape
        
        for i in range(self.num_layers):
            x = self.conv_list[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
                else:
                    xs += self.res_list[i*self.num_kernels+j](x)
                    xs = self.norm_list[i*self.num_kernels+j](xs)
            x = xs / self.num_kernels


            x = F.leaky_relu(x, LRELU_SLOPE) 

            x = self.rnn_list[i](x)
        # bs, 256, n_frames, 9
        # x = x.permute(0,3,1,2).reshape(bs, 2*self.channels[-1], n_frames)
        x = self.conv_post(x) # bs, bottleneck*2, nframes
        
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        
if __name__=='__main__':
    bottleneck_dim=32

    encoder = MixtureEncoder(bottleneck_dim=bottleneck_dim).cuda()
    input = torch.randn(2, 1, 16000).cuda()
    emb = encoder(input)
    print(emb.shape)
    
    x = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(x/1000000)
    

    # rnn = RNNLayer(64, 64).cuda()
    # input = torch.randn(2, 64, 101, 159).cuda()
    # out = rnn(input)
    # print(out.shape)
    # for n, p in encoder.named_parameters():
    #     print(n, p.name, p.data.shape)

