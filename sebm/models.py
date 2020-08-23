import math
from sebm.nets import SimpleNet, Wide_Residual_Net
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class EBM(nn.Module):
    """
    standard EBM without latent variable z
    """
    def __init__(self, arch, **kwargs):
        super().__init__()
        if arch == 'simplenet':
            self.ebm_net = SimpleNet(**kwargs)
        else:
            raise NotImplementError # will implement wresnet-28-10 later
    
    def forward(self, x):
        return self.ebm_net(x)
    
    def energy(self, x):
        return self.forward(x).squeeze()

class CEBM_1ss(nn.Module):
    """
    conjugate EBM with latent variable z,
    where the ebm encodes each image into one 
    neural sufficient statistics w.r.t. natural parameter 1.
    """
    def __init__(self, arch, optimize_priors, device, **kwargs):
        super().__init__()
        if arch == 'simplenet':
            self.ebm_net = SimpleNet(**kwargs)
        else:
            raise NotImplementError # will implement wresnet-28-10 later
            
        self.prior_nat1 = torch.zeros(kwargs['latent_dim']).cuda().to(device)
        self.prior_nat2 = - 0.5 * torch.ones(kwargs['latent_dim']).cuda().to(device) # same prior for each pixel        
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
            
    def forward(self, x):
        return self.ebm_net(x)
    
    def energy(self, x):
        """
        compute the energy function w.r.t. either data distribution 
        or model distribution
        that is defined as
        logA(\lambda) - logA(t(x) + \lambda)
        Ex of the size B         
        """
        neural_ss1 = self.forward(x)
        logA_prior = self.log_partition(self.prior_nat1, self.prior_nat2)
        logA_posterior = self.log_partition(self.prior_nat1+neural_ss1, self.prior_nat2)
        assert logA_prior.shape == (neural_ss1.shape[1],), 'unexpected shape.'
        assert logA_posterior.shape == (neural_ss1.shape[0], neural_ss1.shape[1]), 'unexpected shape.'
        return logA_prior.sum(0) - logA_posterior.sum(1)   
    
    def sample_z_prior(self, sample_size, batch_size):
        """
        return samples from prior of size S * B * latent_dim
        and log_prob of size S * B
        """
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma)       
        latents = prior_dist.sample((sample_size, batch_size, ))
        return latents, prior_dist.log_prob(latents).sum(-1)   
    
    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
    def log_factor(self, neural_ss1, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size  B 
        """
        B, latent_dim = neural_ss1.shape
        assert latents.shape == (B, latent_dim), 'ERROR!'
        return (neural_ss1 * latents).sum(1)
    
class CEBM_2ss(nn.Module):
    """
    conjugate EBM with latent variable z,
    where the ebm encodes each image into two 
    neural sufficient statistics tx1 tx2
    where tx2 = - tx1^2
    """
    def __init__(self, arch, optimize_priors, device, **kwargs):
        super().__init__()
        if arch == 'simplenet':
            self.ebm_net = SimpleNet(**kwargs)
        elif arch =='wresnet':
            self.ebm_net = Wide_Residual_Net(**kwargs)
        else:
            raise NotImplementError # will implement wresnet-28-10 later
            
        self.prior_nat1 = torch.zeros(kwargs['latent_dim']).cuda().to(device)
        self.prior_nat2 = - 0.5 * torch.ones(kwargs['latent_dim']).cuda().to(device)  # same prior for each pixel       
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
            
    def forward(self, x):
        neural_ss1 = self.ebm_net(x)
        neural_ss2 = - neural_ss1**2
        return neural_ss1, neural_ss2
    
    def energy(self, x):
        """
        compute the energy function w.r.t. either data distribution 
        or model distribution
        that is defined as
        logA(\lambda) - logA(t(x) + \lambda)
        Ex of the size B 
        
        argument: dist = 'data' or 'ebm'
        """
        neural_ss1, neural_ss2 = self.forward(x)
        logA_prior = self.log_partition(self.prior_nat1, self.prior_nat2)
        logA_posterior = self.log_partition(self.prior_nat1+neural_ss1, self.prior_nat2+neural_ss2)
        assert logA_prior.shape == (neural_ss1.shape[1],), 'unexpected shape.'
        assert logA_posterior.shape == (neural_ss1.shape[0], neural_ss1.shape[1]), 'unexpected shape.'
        return logA_prior.sum(0) - logA_posterior.sum(1)   
    
    def sample_z_prior(self, sample_size, batch_size):
        """
        return samples from prior of size S * B * latent_dim
        and log_prob of size S * B
        """
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma)       
        latents = prior_dist.sample((sample_size, batch_size, ))
        return latents, prior_dist.log_prob(latents).sum(-1)   
    
    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
    def log_factor(self, neural_ss1, neural_ss2, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size  B 
        """
        B, latent_dim = neural_ss1.shape
        assert latents.shape == (B, latent_dim), 'ERROR!'
        return (neural_ss1 * latents).sum(1) + (neural_ss2 * (latents**2)).sum(1)
    

from sebm.gaussian_params import nats_to_params, params_to_nats

class CEBM_test(nn.Module):
    """
    conjugate EBM with latent variable z,
    where the ebm encodes each image into two 
    neural sufficient statistics tx1 tx2
    where tx2 = - tx1^2
    """
    def __init__(self, arch, optimize_priors, device, reparameterized, **kwargs):
        super().__init__()
        if arch == 'simplenet':
            self.ebm_net = SimpleNet(**kwargs)
        elif arch =='wresnet':
            self.ebm_net = Wide_Residual_Net(**kwargs)
        else:
            raise NotImplementError # will implement wresnet-28-10 later
            
        self.prior_nat1 = torch.zeros(kwargs['latent_dim']).cuda().to(device)
        self.prior_nat2 = - 0.5 * torch.ones(kwargs['latent_dim']).cuda().to(device)  # same prior for each pixel       
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
            
        self.reparameterized = reparameterized
        
    def forward(self, x):
        neural_ss1 = self.ebm_net(x)
        neural_ss2 = - neural_ss1**2
        return neural_ss1, neural_ss2
    
    def energy(self, x):
        """
        compute the energy function w.r.t. either data distribution 
        or model distribution
        that is defined as
        logA(\lambda) - logA(t(x) + \lambda)
        Ex of the size B 
        
        argument: dist = 'data' or 'ebm'
        """
        neural_ss1, neural_ss2 = self.forward(x)
        logA_prior = self.log_partition(self.prior_nat1, self.prior_nat2)
        logA_posterior = self.log_partition(self.prior_nat1+neural_ss1, self.prior_nat2+neural_ss2)
        assert logA_prior.shape == (neural_ss1.shape[1],), 'unexpected shape.'
        assert logA_posterior.shape == (neural_ss1.shape[0], neural_ss1.shape[1]), 'unexpected shape.'
        return logA_prior.sum(0) - logA_posterior.sum(1)   
    
    def latent_prior(self, sample_size, batch_size, latents=None):
        """
        return samples from prior of size S * B * latent_dim
        and log_prob of size S * B
        """
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma) 
        if latents is None:
            if self.reparameterized:
                latents = prior_dist.rsample((sample_size, batch_size,))
            else:
                latents = prior_dist.sample((sample_size, batch_size, ))
        return latents, prior_dist.log_prob(latents).sum(-1)
    
    def latent_posterior(self, sample_size, neural_ss1, neural_ss2, latents=None):
        """
        return samples from the conjugate posterior
        
        """
        posterior_mu, posterior_sigma = nats_to_params(self.prior_nat1+neural_ss1, self.prior_nat2+neural_ss2)
        posterior_dist = Normal(posterior_mu, posterior_sigma) 
        if latents is None:
            if self.reparameterized:
                latents = posterior_dist.rsample((sample_size, ))
            else:
                latents = posterior_dist.sample((sample_size, ))
        return latents, posterior_dist.log_prob(latents).sum(-1)
    
    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
    def log_factor(self, neural_ss1, neural_ss2, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size  S * B 
        """
#         B, latent_dim = neural_ss1.shape
#         assert latents.shape == (B, latent_dim), 'ERROR!'
        return (neural_ss1 * latents).sum(-1) + (neural_ss2 * (latents**2)).sum(-1)
    