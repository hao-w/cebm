import torch
from torch.distributions.normal import Normal

class DATA_NOISE_sampler():
    """
    An sampler for the data noise
    """
    def __init__(self, init_sample_std, noise_std, CUDA, DEVICE):
        super(self.__class__, self).__init__()


        self.noise_dist = Normal(torch.zeros(1).cuda().to(DEVICE),
                                   torch.ones(1).cuda().to(DEVICE) * noise_std)
        
        self.initial_dist = Normal(torch.zeros(1).cuda().to(DEVICE),
                                   torch.ones(1).cuda().to(DEVICE) * init_sample_std)
            
        self.persistent_samples = None
    
    def sgld_update(self, images, num_steps, step_size, persistent=True):
        """
        perform update using slgd
        """
        sample_size, batch_size, pixels_size = images.shape

        if persistent: # persistent sgld
            if self.persistent_samples is None: # if not yet initialized
                self.persistent_samples = self.init_samples(sample_size, batch_size, pixels_size)
            samples = self.persistent_samples

        else: 
            samples = self.init_samples(sample_size, batch_size, pixels_size)

            
        for l in range(num_steps):
            # compute gradient 
            noise = self.noise_dist.sample((sample_size, batch_size, pixels_size,)).squeeze(-1)
            samples = samples - (step_size / 2) * grad_Ex + noise
        if persistent:
            self.persistent_samples = samples
        return samples
        
    def init_samples(self, sample_size, batch_size, pixels_size):
        return self.initial_dist.sample((sample_size, batch_size, pixels_size,)).squeeze(-1) # S * B * pixels
