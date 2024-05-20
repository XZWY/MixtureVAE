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

class ResBlock2d(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock2d, self).__init__()
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

class FEncoder(torch.nn.Module):
    def __init__(self, channels=[2, 64, 64, 64, 64, 64, 64], bottleneck_dim=32):
        super(FEncoder, self).__init__()
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
        
        self.num_kernels = len(resblock_kernel_sizes)
        for c_idx in range(self.num_layers):
            conv_list.append(
                nn.Conv2d(channels[c_idx], channels[c_idx+1], (3, f_kernel_size[c_idx]), stride=(1, f_stride_size[c_idx]), padding=(1,0)),
            )
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes))
                )
            ):
                res_list.append(ResBlock2d(channels[c_idx+1], k, d))    
                norm_list.append(nn.GroupNorm(1, channels[c_idx+1], eps=1e-6, affine=True))
       
        self.conv_list = nn.ModuleList(conv_list)
        self.norm_list = nn.ModuleList(norm_list)
        self.res_list = nn.ModuleList(res_list)
        
        self.conv_list.apply(init_weights)
        
        # self.lstm = nn.LSTM(512, 512, num_layers=2) # bs, T, 512
        
        self.conv_post = weight_norm(nn.Conv2d(channels[-1], bottleneck_dim*2, 1, 1, padding=0))
        # self.conv_post.apply(init_weights)
        
        self.window = torch.hann_window(640)