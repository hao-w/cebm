import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sebm.util import cnn_output_shape, dcnn_output_shape, wres_block_params
import numpy as np
import math

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
        
def _cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, leak=0.01):
    """
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        assert isinstance(leak, (float, int)), "type of leak: %s, expected type: float or int." % type(leak)
        act = nn.LeakyReLU(negative_slope=leak, inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        layers.append(act)
        in_c = out_c
    out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
    flatten_output_dim = out_h * out_w * channels[-1]
    return nn.Sequential(*layers), flatten_output_dim

def _dcnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, leak=0.01, last_act=False):
    """
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        assert isinstance(leak, (float, int)), "type of leak: %s, expected type: float or int." % type(leak)
        act = nn.LeakyReLU(negative_slope=leak, inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        layers.append(act)
        in_c = out_c
    if not last_act:
        layers = layers[:-1]
    return nn.Sequential(*layers)

def _mlp_block(input_dim, hidden_dim, latent_dim, activation, leak=0.01, last_act=False):
    """
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        assert isinstance(leak, (float, int)), "type of leak: %s, expected type: float or int." % type(leak)
        act = nn.LeakyReLU(negative_slope=leak, inplace=True)
    else:
        act = getattr(nn, activation)()
    layers = []
#     last_layer = []
    in_dim = input_dim
    for i, out_dim in enumerate(hidden_dim):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act)
        in_dim = out_dim
    if latent_dim is not None:
        layers.append(nn.Linear(in_dim, latent_dim))
    if last_act:
        layers.append(act)
    return nn.Sequential(*layers) #, nn.Sequential(*last_layer)
    

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    
class SimpleNet(nn.Module):
    """
    Implementation of a cnn-mlp based network
    """
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dim, latent_dim, activation, leak=0.01):
        super().__init__()
        self.cnn_block, self.mlp_input_dim = _cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, leak=leak)
        self.flatten = nn.Flatten()
        self.mlp_block = _mlp_block(self.mlp_input_dim, hidden_dim, latent_dim, activation, leak=leak)
    def forward(self, x):
        h = self.cnn_block(x)
        return self.mlp_block(self.flatten(h))
    
class SimpleNet2(nn.Module):
    """
    Implementation of a cnn-mlp based network
    """
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dim, latent_dim, activation, leak=0.01):
        super().__init__()
        self.cnn_block, self.mlp_input_dim = _cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, leak=leak)
        self.flatten = nn.Flatten()
        self.mlp_block1 = _mlp_block(self.mlp_input_dim, hidden_dim, latent_dim, activation, leak=leak)
        self.mlp_block2 = _mlp_block(self.mlp_input_dim, hidden_dim, latent_dim, activation, leak=leak)
    def forward(self, x):
        h = self.flatten(self.cnn_block(x))
        return self.mlp_block1(h), self.mlp_block2(h)

class SimpleNet3(nn.Module):
    """
    Implementation of a mlp-cnn based network that reverse the encoding procedure of SimpleNet2
    """
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, mlp_input_dim, hidden_dim, mlp_output_dim, activation, leak=0.01):
        super().__init__()
        channels.reverse()
        kernels.reverse()
        strides.reverse()
        paddings.reverse()
        hidden_dim.reverse()
        self.input_channels = channels[0]
        self.channels = channels[1:] + [input_channels]
        self.mlp_block = _mlp_block(mlp_input_dim, hidden_dim, mlp_output_dim, activation, leak=leak, last_act=True)
        self.input_h = int(math.sqrt(mlp_output_dim / self.input_channels))
        self.dcnn_block = _dcnn_block(self.input_h, self.input_h, self.input_channels, self.channels, kernels, strides, paddings, activation, leak=leak)       
    def forward(self, x):
        h = self.mlp_block(x)
        h =h.view(h.shape[0], self.input_channels, self.input_h, self.input_h)
        return self.dcnn_block(h)
             
class MLPNet2(nn.Module):
    """
    Implementation of a mlp based network
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation):
        super().__init__()
        self.mlp_block1 = _mlp_block(input_dim, hidden_dim1, latent_dim=None, activation=activation)
        self.mlp_block2 = _mlp_block(hidden_dim1[-1], hidden_dim2, latent_dim=output_dim, activation=activation)
        self.mlp_block3 = _mlp_block(hidden_dim1[-1], hidden_dim2, latent_dim=output_dim, activation=activation)

    def forward(self, x):
        h = self.mlp_block1(x)
        return self.mlp_block2(h), self.mlp_block3(h)
    
class MLPNet(nn.Module):
    """
    Implementation of a mlp based network
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation):
        super().__init__()
        hidden_dim1.reverse()
        hidden_dim2.reverse()
        self.mlp_block = _mlp_block(input_dim, hidden_dim2+hidden_dim1, output_dim, activation)
    def forward(self, x):
        return self.mlp_block(x)
    
    
class Wres_Block(nn.Module):
    """
    residual block module
        stride   -- stride of the 1st cnn (2nd cnn if swap_cnn=True) in the 1st block in a group

        swap_cnn -- if False, archiecture is the original wresnet block
                      if True,  architecture is the JEM block
                      
        bn_flag -- whether do batch normalization
    """
    def __init__(self, in_c, out_c, stride, activation, dropout_rate=0.0, leak=0.01, swap_cnn=False, bn_flag=False):
        super(Wres_Block, self).__init__()

        self.activation = activation
        
        if bn_flag:
            self.bn1 = nn.BatchNorm2d(in_c, momentum=0.9)
            self.bn2 = nn.BatchNorm2d(out_c, momentum=0.9)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)

        if swap_cnn:
            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=True)
            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)

        if in_c != out_c:
            if bn_flag:
                self.shortcut = nn.Sequential(
                                    nn.BatchNorm2d(in_c, momentum=0.9),
                                    self.activation,
                                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True))
            else:
                self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        h1 = self.dropout(self.conv1(self.activation(self.bn1(x))))
        h2 = self.conv2(self.activation(self.bn2(h1)))
        out = h2 + self.shortcut(x)
        
        return out         
    
class Wide_Residual_Net(nn.Module):
    """
        Implementation of Wide Residual Network https://arxiv.org/pdf/1605.07146.pdf
    """
    def __init__(self, depth, width, im_height=32, im_width=32, input_channels=3, num_classes=10,
                  activation='LeakyReLU', hidden_dim=[10240, 1024], latent_dim=128, dropout_rate=0.0, leak=0.01, swap_cnn=False, bn_flag=False, start_act=True, sum_pool=False):
        super(Wide_Residual_Net, self).__init__()

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        widths = [16] + [int(v * width) for v in (16, 32, 64)]
        print('WRESNET-%d-%d' %(depth, width))

        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(leak)
        elif activation == 'Swish':
            self.activation = Swish()
        else:
            self.activation = getattr(nn, activation)()
        self.dropout_rate = dropout_rate
        self.leak = leak
        self.swap_cnn = swap_cnn
        self.bn_flag = bn_flag
        self.start_act = start_act
        self.sum_pool = sum_pool
        
        conv_params = {
            'kernels' : [],
            'strides' : [],
            'paddings' : []
        }
        self.group1, conv_params = self._init_group(input_channels, widths[0], conv_params)
        self.group2, conv_params = self._wres_group(n, widths[0], widths[1], 1, conv_params)
        self.group3, conv_params = self._wres_group(n, widths[1], widths[2], 2, conv_params)
        self.group4, conv_params = self._wres_group(n, widths[2], widths[3], 2, conv_params)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, 
                                        im_width, 
                                        kernels=conv_params['kernels'], 
                                        strides=conv_params['strides'], 
                                        paddings=conv_params['paddings'])
        
        self.flatten_output_dim = out_h * out_w * widths[3]
        
        self.mlp_block1 = _mlp_block(self.flatten_output_dim, hidden_dim, latent_dim, activation, leak)
        self.mlp_block2 = _mlp_block(self.flatten_output_dim, hidden_dim, latent_dim, activation, leak)

    def _wres_group(self, num_blocks, in_c, out_c, stride, conv_params):
        blocks = []
        for b in range(num_blocks):
            blocks.append(Wres_Block(in_c=(in_c if b == 0 else out_c), 
                          out_c=out_c, 
                          stride=(stride if b == 0 else 1), 
                          activation=self.activation,
                          dropout_rate=self.dropout_rate, 
                          leak=self.leak, 
                          swap_cnn=self.swap_cnn, 
                          bn_flag=self.bn_flag))
            k, s, p = wres_block_params(stride=(stride if b == 0 else 1), swap_cnn=self.swap_cnn)
            conv_params['kernels'] = conv_params['kernels'] + k
            conv_params['strides'] = conv_params['strides'] + s
            conv_params['paddings'] = conv_params['paddings'] + p
        return nn.Sequential(*blocks), conv_params

    def _init_group(self, in_c, out_c, conv_params):
        if self.start_act:
            init_group = nn.Sequential(
                self.activation,
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True))
        else:
            init_group =  nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True))
        conv_params['kernels'].append(3)
        conv_params['strides'].append(1)
        conv_params['paddings'].append(1)
        return init_group, conv_params
        

    def forward(self, x):
        h = self.group1(x)
        h = self.group2(h)
        h = self.group3(h)
        h = self.group4(h)
        h = self.flatten(h)
#         return h
#         if self.sum_pool:
#             out = out.view(out.size(0), out.size(1), -1).sum(2)
#         else:
#             out = F.avg_pool2d(out, 8)
        return self.mlp_block1(self.activation(h)), self.mlp_block2(self.activation(h))
