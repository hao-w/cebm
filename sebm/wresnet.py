"""
Implementation of Wide Residual Network https://arxiv.org/pdf/1605.07146.pdf
"""
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    

class Wres_Block(nn.Module):
    """
    residual block module
        stride   -- stride of the 1st cnn (2nd cnn if swap_cnn=True) in the 1st block in a group

        swap_cnn -- if False, archiecture is the original wresnet block
                      if True,  architecture is the JEM block
                      
        bn_flag -- whether do batch normalization
    """
    def __init__(self, in_c, out_c, stride, act, dropout_rate=0.0, leak=0.05, swap_cnn=False, bn_flag=False):
        super(wres_block, self).__init__()

        self.act = act
        
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
                                    self.act,
                                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True))
            else:
                self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = Identity()

    def forward(self, x):
        h1 = self.dropout(self.conv1(self.act(self.bn1(x))))
        h2 = self.conv2(self.act(self.bn2(h1)))
        out = h2 + self.shortcut(x)
        return out         
    
class Wide_Residual_Net(nn.Module):
    def __init__(self, depth, width, num_classes=10, input_channels=3,
                  act='LeakyReLU', dropout_rate=0.0, leak=0.05, swap_cnn=False, bn_flag=False, start_act=True, sum_pool=False):
        super(Wide_Residual_Net, self).__init__()

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        widths = [16] + [int(v * width) for v in (16, 32, 64)]
        print('| WRESNET-%d-%d' %(depth, width))

        if act == 'LeakyReLU':
            self.act = nn.LeakyReLU(leak)
        elif act == 'Swish':
            self.act = Swish()
        else:
            self.act = getattr(nn, act)()
        self.dropout_rate = dropout_rate
        self.leak = leak
        self.swap_cnn = swap_cnn
        self.bn_flag = bn_flag
        self.start_act = start_act
        self.sum_pool = sum_pool
        
        self.group1 = self._init_group(input_channels, widths[0])
        self.group2 = self._wres_group(n, widths[0], widths[1], 1)
        self.group3 = self._wres_group(n, widths[1], widths[2], 2)
        self.group4 = self._wres_group(n, widths[2], widths[3], 2)
        self.flatten = nn.Flatten()
        self.mlp_block = _mlp_block(self.mlp_input_dim, hidden_dim, latent_dim, activation, leak=leak)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wres_group(self, num_blocks, in_c, out_c, stride):
        blocks = []
        for b in num_blocks:
            blocks.append(Wres_Block(in_c=(in_c if b == 0 else out_c), 
                          out_c=out_c, 
                          stride=(stride if b == 0 else 1), 
                          act=self.act,
                          dropout_rate=self.dropout_rate, 
                          leak=self.leak, 
                          swap_cnn=self.swap_cnn, 
                          bn_flag=self.bn_flag))
        return nn.Sequential(*blocks)

    def _init_group(self, in_c, out_c):
        if self.start_act:
            init_group = nn.Sequential(
                self.act,
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True))
        else:
            init_group =  nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True))
        return init_group
        

    def forward(self, x):
        h = self.group1(x)
        h = self.group2(h)
        h = self.group3(h)
        h = self.group4(h)
        h = self.flatten(h)
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            out = F.avg_pool2d(out, 8)
        return self.mlp_block(out = self.act(self.flatten(h)))
