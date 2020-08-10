import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ffebm.gaussian_params import params_to_nats, nats_to_params
import math

class Energy_function(nn.Module):
    """
    An energy based model that assumes a fully-factorized manner
    with multiple layers till the images is encoded into a scalar value
    neural_ss1 : t(x) w.r.t. the first natural parameter
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, CUDA, DEVICE, optimize_priors=False):
        super(self.__class__, self).__init__()
        self.neural_ss1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        
        
        self.prior_nat1 = torch.zeros(out_channel)
        self.prior_nat2 = - 0.5 * torch.ones(out_channel) # same prior for each pixel        
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
        of size B * latent_dim * P * P (data distribution)
        or
        of size S * B * latent_dim * P * P (model distritbution)
        """
        if dist == 'data':
            return self.neural_ss1(images)  
        elif dist == 'ebm':
            S, B, P, _, in_channel, patch_dim2 = images.shape
#             images = (images - 0.5) / 0.5
            patch_dim = int(math.sqrt(patch_dim2))
            images = images.view(S*B*P*P, in_channel, patch_dim2).view(S*B*P*P, in_channel, patch_dim, patch_dim)
            return self.neural_ss1(images).squeeze(-1).squeeze(-1).view(S, B, P, P, -1).permute(0, 1, 4, 2, 3)
        else:
            raise ValueError
        
    def sample_priors(self, sample_size, batch_size, num_patches):
        """
        return samples from prior of size S * B * P * P * latent_dim
        and log_prob of size S * B * patch_size * patch_size
        """
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma)       
        latents = prior_dist.sample((sample_size, batch_size, num_patches, num_patches, ))
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
        Ex of the size B * P * P
        
        argument: dist = 'data' or 'ebm'
        """
        if dist == 'data':
            B, latent_dim, H, W = neural_ss1.shape
        elif dist == 'ebm':
            S, B, latent_dim, H, W = neural_ss1.shape
        else:
            raise ValueError
        prior_nat1 = self.prior_nat1.unsqueeze(-1).repeat(1, H).unsqueeze(-1).repeat(1, 1, W)
        prior_nat2 = self.prior_nat2.unsqueeze(-1).repeat(1, H).unsqueeze(-1).repeat(1, 1, W) # latent_dim * P * P
        posterior_nat1 = prior_nat1 + neural_ss1
        posterior_nat2 = prior_nat2 # latent_dim * P * P      
        logA_prior = self.normal_log_partition(prior_nat1, prior_nat2)
        logA_posterior = self.normal_log_partition(posterior_nat1, posterior_nat2)
        assert logA_prior.shape == (latent_dim, H, W), 'ERROR!'
        if dist == 'data':
            assert logA_posterior.shape == (B, latent_dim, H, W), 'ERROR!'
            return logA_prior.sum(0) - logA_posterior.sum(1)
        if dist == 'ebm':
            assert logA_posterior.shape == (S, B, latent_dim, H, W), 'ERROR!'
            return logA_prior.sum(0) - logA_posterior.sum(2)
    
    def log_factor(self, neural_ss1, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size S * B * H * W
        """
        S, B, latent_dim, H, W = neural_ss1.shape
        assert latents.shape == (S, B, H, W, latent_dim), 'ERROR!'
        return (neural_ss1 * latents.permute(0, 1, 4, 2, 3)).sum(2)