import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ffebm.gaussian_params import params_to_nats, nats_to_params

class Energy_function(nn.Module):
    """
    The Vanilla CNN network used in the Anatomy model
    """
    def __init__(self, latent_dim, CUDA, DEVICE, negative_slope=0.05, optimize_priors=False):
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
        self.prior_nat2 = - 5 * torch.ones(latent_dim) # same prior for each pixel        
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_nat1 = self.prior_nat1.cuda()
                self.prior_nat2 = self.prior_nat2.cuda()  
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
            
    def forward(self, images, latents=None):
        """
        Encode the images and compute the energy function,
        in addition, if the latents is given, also compute the 
        heuristic factor.
        """
        if latents is None:
            B, C, _, _ = images.shape
            neural_ss1 = self.fc(self.cnn(images).view(B, 288)) 
        else:
            S, B, C, pixel_size, _, = images.shape
            neural_ss1 = self.fc(self.cnn(images.view(S*B, 1, pixel_size, pixel_size)).view(S*B, 288)).view(S, B, latents.shape[-1])
#         neural_ss1 = self.fc(self.cnn(images).view(B, 288)) 
        Ex = self.energy(prior_nat1=self.prior_nat1,
                         prior_nat2=self.prior_nat2,
                         posterior_nat1=self.prior_nat1 + neural_ss1,
                         posterior_nat2=self.prior_nat2)
        if latents is None:
            return Ex.sum(-1) # length-B
        else:
            return Ex.sum(-1), (neural_ss1 * latents).sum(-1)
            
    def priors(self, sample_size, batch_size, samples=None):
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma)
        if samples is None:         
            samples = prior_dist.sample((sample_size, batch_size, ))
        log_pdf = prior_dist.log_prob(samples).sum(-1) # size  S * B 
        return samples, log_pdf
            
    def normal_log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()          
    
    def energy(self, prior_nat1, prior_nat2, posterior_nat1, posterior_nat2):
        """
        compute the energy function that is defined as
        logA(\lambda) - logA(t(x) + \lambda)
        """
        logA_prior = self.normal_log_partition(prior_nat1, prior_nat2) 
        logA_posterior = self.normal_log_partition(posterior_nat1, posterior_nat2)
        return logA_prior - logA_posterior
        