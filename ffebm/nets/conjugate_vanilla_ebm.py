import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ffebm.gaussian_params import params_to_nats, nats_to_params
import math

class Energy_function(nn.Module):
    """
    An energy based model
    """
    def __init__(self, latent_dim, CUDA, DEVICE, optimize_priors=False):
        super(self.__class__, self).__init__()
        
        self.cnn1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.cnn3 = nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.cnn4 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)

        self.fc1 = nn.Linear(288, 128)
        self.fc2 = nn.Linear(128, latent_dim)
        
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.01, inplace=True))

#         self.fc = nn.Sequential(
#             nn.Linear(288, 128),
#             nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
#             nn.Linear(128, latent_dim))
        
        self.prior_nat1 = torch.zeros(latent_dim)
        self.prior_nat2 = - 0.5 * torch.ones(latent_dim) # same prior for each pixel        
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_nat1 = self.prior_nat1.cuda()
                self.prior_nat2 = self.prior_nat2.cuda()  
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
            
    def forward(self, images):
        """
        Encode the images into some neural sufficient statistics
        of size B * latent_dim (data distribution)
        or
        of size S * B * latent_dim (model distritbution)
        """
        B = images.shape[0]
        h1 = self.cnn1(images) 
        h2 = self.cnn2((h1 * torch.sigmoid(h1)))
        h3 = self.cnn2((h2 * torch.sigmoid(h2)))
        h4 = self.cnn2((h3 * torch.sigmoid(h3)))
        h4 = h4.view(B, 288)
        h5 = self.fc1(h4 * torch.sigmoid(h4))
        return self.fc2(h5 * torch.sigmoid(h5))
        
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
    
    def energy(self, neural_ss1):
        """
        compute the energy function w.r.t. either data distribution 
        or model distribution
        that is defined as
        logA(\lambda) - logA(t(x) + \lambda)
        Ex of the size B 
        
        argument: dist = 'data' or 'ebm'
        """
        B, latent_dim = neural_ss1.shape
        prior_nat1 = self.prior_nat1
        prior_nat2 = self.prior_nat2 # latent_dim 
        posterior_nat1 = prior_nat1 + neural_ss1
        posterior_nat2 = prior_nat2 # latent_dim     
        logA_prior = self.normal_log_partition(prior_nat1, prior_nat2)
        logA_posterior = self.normal_log_partition(posterior_nat1, posterior_nat2)
        assert logA_prior.shape == (latent_dim,), 'ERROR!'
        assert logA_posterior.shape == (B, latent_dim), 'ERROR!'
        return logA_prior.sum(0) - logA_posterior.sum(1)

    def log_factor(self, neural_ss1, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size  B 
        """
        B, latent_dim = neural_ss1.shape
        assert latents.shape == (B, latent_dim), 'ERROR!'
        return (neural_ss1 * latents).sum(1)