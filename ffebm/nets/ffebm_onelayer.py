import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ffebm.gaussian_params import params_to_nats, nats_to_params
import math

class Energy_function(nn.Module):
    """
    An energy based model
    """
    def __init__(self, latent_dim, CUDA, DEVICE, negative_slope=0.01, optimize_priors=False):
        super(self.__class__, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

        self.fc = nn.Sequential(
            nn.Linear(288, 128),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(128, latent_dim))
        
        
        self.prior_nat1 = torch.zeros(latent_dim)
        self.prior_nat2 = - 0.5 * torch.ones(latent_dim) # same prior for each pixel        
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_nat1 = self.prior_nat1.cuda()
                self.prior_nat2 = self.prior_nat2.cuda()  
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
            
    def forward(self, images, dist=None):
        """
        Encode the images into some neural sufficient statistics
        of size B * latent_dim (data distribution)
        or
        of size S * B * latent_dim (model distritbution)
        """
        if dist == 'data':
            B = images.shape[0]
            h1 = self.cnn(images) 
            return self.fc(h1.view(B, 288)) 
        elif dist == 'ebm':
            S, B, C, P, _ = images.shape
            h1 = self.cnn(images.view(S*B, C, P, P))
            return self.fc(h1.view(S*B, 288)).view(S, B, -1)
            
        else:
            raise ValueError
        
    def sample_priors(self, sample_size, batch_size):
        """
        return samples from prior of size S * B * latent_dim
        and log_prob of size S * B
        """
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma)       
        latents = prior_dist.sample((sample_size, batch_size, ))
        return latents, prior_dist.log_prob(latents).sum(-1)

    def normal_log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()          
    
    def energy(self, neural_ss1, dist=None):
        """
        compute the energy function w.r.t. either data distribution 
        or model distribution
        that is defined as
        logA(\lambda) - logA(t(x) + \lambda)
        Ex of the size B 
        
        argument: dist = 'data' or 'ebm'
        """
        if dist == 'data':
            B, latent_dim = neural_ss1.shape
        elif dist == 'ebm':
            S, B, latent_dim = neural_ss1.shape
        else:
            raise ValueError
        prior_nat1 = self.prior_nat1
        prior_nat2 = self.prior_nat2 # latent_dim 
        posterior_nat1 = prior_nat1 + neural_ss1
        posterior_nat2 = prior_nat2 # latent_dim * P * P      
        logA_prior = self.normal_log_partition(prior_nat1, prior_nat2)
        logA_posterior = self.normal_log_partition(posterior_nat1, posterior_nat2)
        assert logA_prior.shape == (latent_dim,), 'ERROR!'
        if dist == 'data':
            assert logA_posterior.shape == (B, latent_dim), 'ERROR!'
            return logA_prior.sum(0) - logA_posterior.sum(1)
        if dist == 'ebm':
            assert logA_posterior.shape == (S, B, latent_dim), 'ERROR!'
            return logA_prior.sum(0) - logA_posterior.sum(2)
    
    def log_factor(self, neural_ss1, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size S * B 
        """
        S, B, latent_dim = neural_ss1.shape
        assert latents.shape == (S, B, latent_dim), 'ERROR!'
        return (neural_ss1 * latents).sum(2)