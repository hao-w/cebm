import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

class SGLD_sampler():
    """
    An sampler using stochastic gradient langevin dynamics 
    """
    def __init__(self, init_sample_std, noise_std, CUDA, DEVICE):
        super(self.__class__, self).__init__()


        self.noise_dist = Normal(torch.zeros(1).cuda().to(DEVICE),
                                   torch.ones(1).cuda().to(DEVICE) * noise_std)
        
#         self.initial_dist = Normal(torch.zeros(1).cuda().to(DEVICE),
#                                    torch.ones(1).cuda().to(DEVICE) * init_sample_std)
        self.initial_dist = Uniform(-1 * torch.ones(1).cuda().to(DEVICE),
                                   torch.ones(1).cuda().to(DEVICE))
                        
        self.persistent_samples = None
    
    def sgld_update(self, ef, batch_size, pixels_size, num_steps, step_size, buffer_size, buffer_percent, persistent=True):
        """
        perform update using slgd
        """
        
        if persistent: # persistent sgld
            if self.persistent_samples is None: # if not yet initialized
                self.persistent_samples = self.init_samples(buffer_size, pixels_size)
                assert self.persistent_samples.shape == (buffer_size, 1, pixels_size, pixels_size), "ERROR! buffer sampels have expected shape."
            num_from_buffer = int(buffer_percent * batch_size)
            num_from_init = batch_size - num_from_buffer
            samples_from_buffer = self.sample_from_buffer(num_from_buffer)
            
            samples_from_init = self.init_samples(num_from_init, pixels_size)
            samples = torch.cat((samples_from_buffer, samples_from_init), 0)
            assert samples.shape == (batch_size, 1, pixels_size, pixels_size), "ERROR! samples have unexpected shape."
    
        else: 
            samples = self.init_samples(batch_size, pixels_size)

            
        for l in range(num_steps):
            # compute gradient 
            samples.requires_grad = True
            grads_tuple = torch.autograd.grad(outputs=ef.forward(samples).sum(), inputs=samples)
            grads = torch.clamp(grads_tuple[0], min=-1e-2, max=1e-2)
            noise = self.noise_dist.sample((batch_size, 1, pixels_size, pixels_size,)).squeeze(-1)
            samples = (samples - (step_size / 2) * grads + noise).detach()
        if persistent:
            self.persistent_samples = torch.cat((samples[:num_from_buffer], self.persistent_samples[num_from_buffer:]), 0)
            assert self.persistent_samples.shape == (buffer_size, 1, pixels_size, pixels_size), "ERROR! buffer samples have unexpected shape."
        return samples
        
    def init_samples(self, batch_size, pixels_size):
        return self.initial_dist.sample((batch_size, 1, pixels_size, pixels_size,)).squeeze(-1)
    
    def sample_from_buffer(self, sample_size):
        rand_index = torch.randperm(self.persistent_samples.shape[0])
        self.persistent_samples = self.persistent_samples[rand_index]
        return self.persistent_samples[:sample_size]
        
