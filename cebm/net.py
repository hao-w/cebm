import os
import math
import torch
import torch.nn as nn
import numpy as np
                 
def cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, **kwargs):
    """
    building blocks for a convnet.
    each block is in form of:
        Conv2d
        BatchNorm2d(optinal)
        Activation
        Dropout(optional)
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        if (i < (len(channels)-1)) or last_act:#Last layer will be customized 
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(act)
            if 'dropout_prob' in kwargs:
                layers.append(nn.Dropout2d(kwargs['dropout_prob']))
            if 'maxpool_kernels' in kwargs and 'maxpool_strides' in kwargs:
                layers.append(nn.MaxPool2d(kernel_size=kwargs['maxpool_kernels'][i], stride=kwargs['maxpool_strides'][i]))
        in_c = out_c
    return nn.Sequential(*layers)

def deconv_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act, batchnorm, **kwargs):
    """
    building blocks for a deconvnet
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    assert len(channels) == len(kernels), "length of channels: %s,  length of kernels: %s" % (len(channels), len(kernels))
    assert len(channels) == len(strides), "length of channels: %s,  length of strides: %s" % (len(channels), len(strides))
    assert len(channels) == len(paddings), "length of channels: %s,  length of kernels: %s" % (len(channels), len(paddings))
    layers = []
    in_c = input_channels
    for i, out_c in enumerate(channels):
        layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=kernels[i], stride=strides[i], padding=paddings[i]))
        if (i < (len(channels)-1)) or last_act:
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_c)) 
            layers.append(act)
        in_c = out_c
    return nn.Sequential(*layers)

def mlp_block(input_dim, hidden_dims, activation, **kwargs):
    """
    building blocks for a mlp
    """
    if activation == 'Swish':
        act = Swish()
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU(negative_slope=kwargs['leak_slope'], inplace=True)
    else:
        act = getattr(nn, activation)()
    layers = []
    in_dim = input_dim
    for i, out_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act)
        in_dim = out_dim
    return nn.Sequential(*layers)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)

class Swish(nn.Module):
    """
    The swish activation function
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
    
def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)

    h = (h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1

    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of deconvolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(padding) is not tuple:
        padding = (padding, padding)
    h = (h_w[0] - 1) * stride[0] - 2 * padding[0]  + (dilation * (kernel_size[0] - 1)) + 1
    w = (h_w[1] - 1) * stride[1] - 2 * padding[1]  + (dilation * (kernel_size[1] - 1)) + 1

    return h, w

def cnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = conv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w

def dcnn_output_shape(h, w, kernels, strides, paddings):
    h_w = (h, w)
    for i, kernel in enumerate(kernels):
        h_w = deconv_output_shape(h_w, kernels[i], strides[i], paddings[i])
    return h_w

########################################################
####################Residual Network####################
########################################################
def wres_block_params(stride, swap_cnn):
    kernels = [3,3]
    paddings = [1,1]
    if swap_cnn:
        strides = [1, stride]
    else:
        strides = [stride, 1]
    return kernels, strides, paddings
        
    
class BasicBlock(nn.Module):
    """
    basic block module
        stride   -- stride of the 1st cnn in the 1st block in a group
        bn_flag -- whether do batch normalization
    """
    def __init__(self, in_c, out_c, stride, activation, bn_flag=False):
        super(Wres_Block, self).__init__()

        self.activation = activation
        
        if bn_flag:
            self.bn1 = nn.BatchNorm2d(in_c, momentum=0.9)
            self.bn2 = nn.BatchNorm2d(out_c, momentum=0.9)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=True)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)

        if in_c != out_c:
            if bn_flag:
                self.shortcut = nn.Sequential(
                                    nn.BatchNorm2d(in_c, momentum=0.9),
                                    activation,
                                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True))
            else:
                self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = nn.Identity()

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
