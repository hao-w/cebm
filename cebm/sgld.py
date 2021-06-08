import time
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


class SGLD_Sampler():
    """
    An sampler using stochastic gradient langevin dynamics 
    """
    def __init__(self, im_h, im_w, im_channels, device, alpha, noise_std, buffer_size, reuse_freq):
        super().__init__()
        im_dims = (im_channels, im_h, im_w)
        self.initial_dist = Uniform(-1 * torch.ones(im_dims).to(device), torch.ones(im_dims).to(device))
        self.device = device
        self.alpha = alpha
        self.noise_std = noise_std
        self.reuse_freq = reuse_freq
        self.buffer = self.initial_dist.sample((buffer_size, ))

    def sample_from_buffer(self, batch_size):
        """
        sample from buffer with a frequency
        self.buffer_dup_allowed = True allows sampling the same chain multiple time within one sampling step
        which is used in JEM and IGEBM  
        """
        samples = self.initial_dist.sample((batch_size, ))
        inds = torch.randint(0, len(self.buffer), (batch_size, ), device=self.device)
        samples_from_buffer = self.buffer[inds]
        rand_mask = (torch.rand(batch_size, device=self.device) < self.reuse_freq)
        samples[rand_mask] = samples_from_buffer[rand_mask]
        return samples, inds
    
    def refine_buffer(self, samples, inds):
        """
        update replay buffer
        """
        self.buffer[inds] = samples

    def sample(self, ebm, batch_size, num_steps, pcd=True, init_samples=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples, inds = self.sample_from_buffer(batch_size)
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.energy(samples.tanh()).sum(), inputs=samples)[0]
            samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            #added this extra detachment step, becase the last update keeps the variable in the graph.
            samples = samples.detach() 
        assert samples.requires_grad == False, "samples should not require gradient."
        if pcd:
            self.refine_buffer(samples.detach(), inds)
        return samples
    
    def cond_sample(self, ebm, z, batch_size, num_steps, pcd=True, init_samples=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples, inds = self.sample_from_buffer(batch_size)
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.log_factor(samples, z).sum(), inputs=samples)[0]
            samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            #added this extra detachment step, becase the last update keeps the variable in the graph.
            samples = samples.detach() 
        assert samples.requires_grad == False, "samples should not require gradient."
        if pcd:
            self.refine_buffer(samples.detach(), inds)
        return samples
    

class SGLD_Sampler_GMM():
    """
    sgld sampler for meta learning  
    """
    def __init__(self, im_h, im_w, im_channels, device, alpha, noise_std, buffer_size, reuse_freq):
        super().__init__()
        im_dims = (im_channels, im_h, im_w)
        self.initial_dist = Uniform(-1 * torch.ones(im_dims).to(device), torch.ones(im_dims).to(device))
        self.device = device
        self.alpha = alpha
        self.noise_std = noise_std
        self.reuse_freq = reuse_freq
        self.buffer = self.initial_dist.sample((buffer_size, ))

    def sample_from_buffer(self, batch_size):
        """
        sample from buffer with a frequency
        self.buffer_dup_allowed = True allows sampling the same chain multiple time within one sampling step
        which is used in JEM and IGEBM  
        """
        samples = self.initial_dist.sample((batch_size, ))
        inds = torch.randint(0, len(self.buffer), (batch_size, ), device=self.device)
        samples_from_buffer = self.buffer[inds]
        rand_mask = (torch.rand(batch_size, device=self.device) < self.reuse_freq)
        samples[rand_mask] = samples_from_buffer[rand_mask]
        return samples, inds
    
    def refine_buffer(self, samples, inds):
        """
        update replay buffer
        """
        self.buffer[inds] = samples

    def sample(self, ebm, batch_size, num_steps, c_means, c_stds, ys, pcd=True, init_samples=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples, inds = self.sample_from_buffer(batch_size)
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.energy(samples, c_means, c_stds, ys).sum(), inputs=samples)[0]
            samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            #added this extra detachment step, becase the last update keeps the variable in the graph.
            samples = samples.detach() 
        assert samples.requires_grad == False, "samples should not require gradient."
        if pcd:
            self.refine_buffer(samples.detach(), inds)
        return samples
