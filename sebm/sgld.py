import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import time

class SGLD_sampler():
    """
    An sampler using stochastic gradient langevin dynamics 
    """
    def __init__(self, device, input_channels, noise_std, lr, pixel_size, buffer_size, buffer_percent, buffer_init, buffer_dup_allowed, grad_clipping=False):
        super(self.__class__, self).__init__()
        self.initial_dist = Uniform(-1 * torch.ones((input_channels, pixel_size, pixel_size)).cuda().to(device),
                                   torch.ones((input_channels, pixel_size, pixel_size)).cuda().to(device))
          
        self.lr = lr
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.buffer_percent = buffer_percent
        self.grad_clipping=grad_clipping
        self.buffer_init = buffer_init
        if self.buffer_init: # whether initialize buffer at the beginning 
            self.buffer = self.initial_dist.sample((self.buffer_size, ))
            self.buffer_dup_allowed = buffer_dup_allowed
        else:
            self.buffer = None
            self.buffer_dup_allowed = True # without init buffer, always allow duplicated sampling, i.e. ignore that parameter    
        self.device = device

    def sample_from_buffer(self, batch_size):
        """
        sample from buffer with a frequency
        self.buffer_dup_allowed = True allows sampling the same chain multiple time within one sampling step
        which is used in JEM and IGEBM  
        """
        if self.buffer_dup_allowed:
            samples = self.initial_dist.sample((batch_size, ))
            inds = torch.randint(0, self.buffer_size, (batch_size, ), device=self.device)
            samples_from_buffer = self.buffer[inds]
            rand_mask = (torch.rand(batch_size, device=self.device) < self.buffer_percent)
            samples[rand_mask] = samples_from_buffer[rand_mask]
        else:
            inds = int(self.buffer_percent * batch_size)
            self.buffer = self.buffer[torch.randperm(len(self.buffer))]
            samples_from_buffer = self.buffer[:inds]
            samples_from_init = self.initial_dist.sample((batch_size - inds,))
            samples = torch.cat((samples_from_buffer, samples_from_init), 0)
        assert samples.shape[0] == batch_size, "Samples have unexpected shape."            
        return samples, inds

    def nsgd_steps(self, ebm, samples, num_steps, logging_interval=None):
        """
        perform noisy gradient descent steps and return updated samples 
        """
        list_samples = []
        
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.energy(samples).sum(), inputs=samples)[0]
            if self.grad_clipping:
                grads = torch.clamp(grads, min=-1e-2, max=1e-2)
            samples = (samples - (self.lr / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            if logging_interval is not None:
                if (l+1) % logging_interval == 0:
                    list_samples.append(samples.unsqueeze(0).detach())  
        if logging_interval is not None:
            samples = torch.cat(list_samples, 0)
        else:
            samples = samples.detach() ## added this extra detachment step, becase the last update keeps the variable in the graph somehow, need to figure out why.
        assert samples.requires_grad == False, "samples should not require gradient."
        return samples
    
    def refine_buffer(self, samples, inds):
        """
        update replay buffer
        """
        if self.buffer_init:
            if self.buffer_dup_allowed:
                self.buffer[inds] = samples
            else:
                self.buffer[:inds] = samples[:inds]
        else:
            if self.buffer is None:
                self.buffer = samples
            else:
                self.buffer = torch.cat((self.buffer, samples), 0)
                if self.buffer_size < len(self.buffer):
                    ## truncate buffer from 'head' of the queue
                    self.buffer = self.buffer[len(self.buffer) - self.buffer_size:]


    def nsgd_steps_ll(self, ebm, samples, latents, num_steps, logging_interval=None):
        """
        perform noisy gradient descent steps and return updated samples 
        """
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            ll = ebm.log_factor(samples, latents)
            grads = torch.autograd.grad(outputs=ll.sum(), inputs=samples)[0]
            if self.grad_clipping:
                grads = torch.clamp(grads, min=-1e-2, max=1e-2)
            samples = (samples - (self.lr / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            if logging_interval is not None:
                if (l+1) % logging_interval == 0:
                    list_samples.append(samples.unsqueeze(0).detach())  
        if logging_interval is not None:
            samples = torch.cat(list_samples, 0)
        else:
            samples = samples.detach() ## added this extra detachment step, becase the last update keeps the variable in the graph somehow, need to figure out why.
        assert samples.requires_grad == False, "samples should not require gradient."
        return samples
    
    def sample_ll(self, ebm, latents, batch_size, num_steps, init_samples=None, logging_interval=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        if buffer is not initialized in advance, we check its storage and 
        init from random noise if storage is smaller than batch_size.
        """
        if init_samples is None:
            samples = self.initial_dist.sample((batch_size, ))
        else:
            samples = init_samples
        samples = self.nsgd_steps_ll(ebm, samples, latents, num_steps, logging_interval=logging_interval)
        return samples
    
    def sample(self, ebm, batch_size, num_steps, pcd=True, init_samples=None, logging_interval=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        if buffer is not initialized in advance, we check its storage and 
        init from random noise if storage is smaller than batch_size.
        """
        if pcd:
            if self.buffer_init:
                samples, inds = self.sample_from_buffer(batch_size)
            else:
                if self.buffer is not None and len(self.buffer) >= batch_size:     
                    samples, _ = self.sample_from_buffer(batch_size)
                else: 
                    samples = self.initial_dist.sample((batch_size, ))
                inds = None
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        samples = self.nsgd_steps(ebm, samples, num_steps, logging_interval=logging_interval)
        ## refine buffer if pcd
        if pcd:
            self.refine_buffer(samples.detach(), inds)
        return samples
    
    def nsgd_steps_cond(self, class_label, ebm, samples, num_steps, logging_interval=None):
        """
        perform noisy gradient descent steps and return updated samples 
        """
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.energy_cond(samples, class_label).sum(), inputs=samples)[0]
            if self.grad_clipping:
                grads = torch.clamp(grads, min=-1e-2, max=1e-2)
            samples = (samples - (self.lr / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            if logging_interval is not None:
                if (l+1) % logging_interval == 0:
                    list_samples.append(samples.unsqueeze(0).detach())  
        if logging_interval is not None:
            samples = torch.cat(list_samples, 0)
        else:
            samples = samples.detach() ## added this extra detachment step, becase the last update keeps the variable in the graph somehow, need to figure out why.
        assert samples.requires_grad == False, "samples should not require gradient."
        return samples
    
    def sample_cond(self, class_label, ebm, batch_size, num_steps, pcd=True, init_samples=None, logging_interval=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        if buffer is not initialized in advance, we check its storage and 
        init from random noise if storage is smaller than batch_size.
        """
        if pcd:
            if self.buffer_init:
                samples, inds = self.sample_from_buffer(batch_size)
            else:
                if self.buffer is not None and len(self.buffer) >= batch_size:     
                    samples, _ = self.sample_from_buffer(batch_size)
                else: 
                    samples = self.initial_dist.sample((batch_size, ))
                inds = None
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        samples = self.nsgd_steps_cond(class_label, ebm, samples, num_steps, logging_interval=logging_interval)
        ## refine buffer if pcd
        if pcd:
            self.refine_buffer(samples, inds)
        return samples