import torch
from torch.distributions.normal import Normal
from sebm.gaussian_params import params_to_nats, nats_to_params
import sebm.nets
import math

class EBM(nn.Module):
    """
    """
    def __init__(self, arch, **kwargs):
        super().__init__()
        if arch == 'simplenet':
            
        else:
            raise NotImplementError # will implement wresnet-28-10 later
