import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from quasi_conj.gaussian_conjugacy import params_to_nats, nats_to_params, posterior_nats, unnormalized_marginal_log_prob

class Sampler_images(nn.Module):
    """
    The generative model which is similar to a decoder in a VAE, except we do not sample reconstructed data
    """
    def __init__(self, latents_dim, hidden_dim, pixels_dim, reparameterized, CUDA, DEVICE):
        super(self.__class__, self).__init__()
        self.reconstruct_images = nn.Sequential(
                nn.Linear(latents_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, pixels_dim),
                nn.Sigmoid())
        
        self.prior_nat1 = torch.zeros(latents_dim)
        self.prior_nat2 = - 0.5 * torch.ones(latents_dim)       
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_nat1 = self.prior_nat1.cuda()
                self.prior_nat2 = self.prior_nat2.cuda()   
                
        self.reparameterized = reparameterized
                
    def forward(self, images, latents):
        p_images = dict()
#         prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
#         prior_dist = Normal(prior_mu, prior_sigma)
#         p_images['log_prior'] = prior_dist.log_prob(latents).sum(-1)
        p_images['image_means'] = self.reconstruct_images(latents)
        bce = self.binary_cross_entropy(p_images['image_means'], images)
        p_images['log_prob'] = - bce
        return p_images
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)
        
    def sample_from_prior(self, sample_size, batch_size):
        p = dict()
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(loc=prior_mu, scale=prior_sigma)
        if self.reparameterized:
            latents = prior_dist.rsample((sample_size, batch_size,))
        else:
            latents = prior_dist.sample((sample_size, batch_size,))
        p['log_prob'] = prior_dist.log_prob(latents).sum(-1)
        p['samples'] = latents
        return p