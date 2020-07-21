import torch
from torch.distributions.normal import Normal

class DATA_NOISE_sampler():
    """
    An sampler for the data noise
    """
    def __init__(self, noise_std, CUDA, DEVICE):
        super(self.__class__, self).__init__()

        self.noise_dist = Normal(torch.zeros(1).cuda().to(DEVICE),
                                   torch.ones(1).cuda().to(DEVICE) * noise_std)
        
    def sample(self, sample_size, batch_size, pixels_size):
        return self.noise_dist.sample((sample_size, batch_size, pixels_size,)).squeeze(-1) # S * B * pixels
