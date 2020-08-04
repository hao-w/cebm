import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from sebm.ffebm.gaussian_params import params_to_nats, nats_to_params

class Energy_function(nn.Module):
    """
    An energy based model that assumes a fully-factorized manner
    neural_ss1 : t(x) w.r.t. the first natural parameter
    """
    def __init__(self, CUDA, DEVICE, in_channel=1, out_channel=32, negative_slope=0.05, optimize_priors=False):
        super(self.__class__, self).__init__()
        self.neural_ss1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope))
        
        
        self.prior_nat1 = torch.zeros(out_channel)
        self.prior_nat2 = - 0.5 * torch.ones(out_channel) # same prior for each pixel        
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.prior_nat1 = self.prior_nat1.cuda()
                self.prior_nat2 = self.prior_nat2.cuda()  
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
            
    def forward(self, images):
        """
        Encode the images and compute the energy function
        """
        neural_ss1 = self.neural_ss1(images)  
        B, C, H, W = neural_ss1.shape
        neural_ss1_flat = neural_ss1.view(B, C, H*W) # B * C * P 
        Ex = self.energy(prior_nat1=self.prior_nat1.repeat(1, H*W),
                         prior_nat2=self.prior_nat2.repeat(1, H*W),
                         posterior_nat1=self.prior_nat1.repeat(1, H*W) + neural_ss1_flat,
                         posterior_nat2=self.prior_nat2.repeat(1, H*W))
        return Ex # B * C * P
        
#     def prior_log_pdf(self, samples=None):
#         prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
#         prior_dist = Normal(prior_mu, prior_sigma)
#         if samples is None:         
#             raise NotImplementedError
#         else:
#             return prior_dist.log_prob(samples) # size  C * P
        
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
        logA_prior = normal_log_partition(prior_nat1, prior_nat2) 
        logA_posterior = normal_log_partition(posterior_nat1, posterior_nat2)
        return logA_prior - logA_posterior
        