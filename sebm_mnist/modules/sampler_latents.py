import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from quasi_conj.gaussian_conjugacy import params_to_nats, nats_to_params, posterior_nats, unnormalized_marginal_log_prob

class Sampler_latents(nn.Module):
    """
    A sampler that can either sample from the prior of latents,
    or sample form the quasi-conjugate posterior
    """
    def __init__(self, pixels_dim, hidden_dim, neural_ss_dim, optimize_priors, reparameterized):
        super(self.__class__, self).__init__()
        
        self.neural_ss = nn.Sequential(
            nn.Linear(pixels_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, neural_ss_dim))
        self.reparameterized = reparameterized
    
    def forward(self, images, prior_nat1, prior_nat2, latents=None):
        q_latents = dict()
        neural_ss = self.neural_ss(images)
        
        posterior_nat1, posterior_nat2 = posterior_nats(prior_nat1, prior_nat2, neural_ss)
        posterior_mu, posterior_sigma = nats_to_params(posterior_nat1, posterior_nat2)
        posterior_dist = Normal(posterior_mu, posterior_sigma)
        if self.reparameterized:
            latents = posterior_dist.rsample()
        else:
            latents = posterior_dist.sample()
        log_prob = posterior_dist.log_prob(latents).sum(-1)
        
        q_latents['samples'] = latents
        q_latents['log_prob'] = log_prob
        return q_latents
    

            
#         if latents is None:
# 
#             posterior_mu, posterior_sigma = self.compute_posterior(neural_ss)
#             latents, log_prob = self.sample_from_posterior(posterior_mu, posterior_sigma)
#             q_latents['posterior_mean'] = posterior_mu
#             q_latents['posterior_std'] = posterior_sigma
#             q_latents['samples'] = latents
#             q_latents['log_prob'] = log_prob
#         log_marginal_images, log_likelihood_images = self.eval_from_posterior(neural_ss, latents)
#         q_latents['log_marginal_images'] = log_marginal_images
#         q_latents['log_likelihood_images'] = log_likelihood_images
#         return q_latents
            
#     def compute_posterior(self, neural_ss):
#         """
#         compute the analytic form of natural parameters of the posterior
#         """
# #         prior_nat1, prior_nat2 = params_to_nats(mu=self.prior_mean, sigma=self.prior_std)
#         posterior_nat1, posterior_nat2 = posterior_nats(self.prior_nat1, self.prior_nat2, neural_ss)
#         posterior_mu, posterior_sigma = nats_to_params(posterior_nat1, posterior_nat2)
#         return posterior_mu, posterior_sigma
    
#     def sample_from_posterior(self, posterior_mu, posterior_sigma):
#         """
#         sample from posterior
#         """
#         q_latents = dict()
#         posterior_dist = Normal(posterior_mu, posterior_sigma)
#         if self.reparameterized:
#             latents = posterior_dist.rsample()
#         else:
#             latents = posterior_dist.sample()
#         log_prob = posterior_dist.log_prob(latents).sum(-1)
#         return latents, log_prob
    
#     def eval_from_posterior(self, neural_ss, latents):
#         """
#         evaluate all necessary densities
#         """
#         prior_nat1, prior_nat2 = params_to_nats(mu=self.prior_mean, sigma=self.prior_std)
#         log_likelihood_images = self.log_likelihood_images(neural_ss, latents)
#         return log_marginal_images, log_likelihood_images

    def log_marginal(self, images, prior_nat1, prior_nat2):
        neural_ss = self.neural_ss(images)
        return unnormalized_marginal_log_prob(prior_nat1, prior_nat2, neural_ss)

    def log_factor(self, images, latents):
        """
        Given neural_ss and latents, evaluate the unnormalized likelihood \gamma(x | z)
        """
        neural_ss = self.neural_ss(images)
        return (neural_ss * latents).sum(-1)
    
    def log_joint(self, images, latents, prior_nat1, prior_nat2):
        prior_mu, prior_sigma = nats_to_params(prior_nat1, prior_nat2)
        prior_dist = Normal(loc=prior_mu, scale=prior_sigma)
        log_prior = prior_dist.log_prob(latents).sum(-1)
        neural_ss = self.neural_ss(images)
        log_factor = (neural_ss * latents).sum(-1) 
        return log_factor + log_prior
