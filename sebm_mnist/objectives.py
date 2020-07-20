import torch.nn.functional as F
from torch import logsumexp
"""
MAP : maximize A posterior w.r.t. phi and lambda, 
EUBO : minimize an upper bound on the KL (p^{DATA} (x) || \pi(x))
"""

def MIN_KL(modules, images, opt_param=None):
    S, B, _ = images.shape
    (sampler_latents, sampler_images) = modules
    log_marginal_from_q = sampler_latents.log_marginal(images, sampler_images.prior_nat1, sampler_images.prior_nat2)
    priors = sampler_images.sample_from_prior(S, B)
    proposals = sampler_images.forward(images, priors['samples'])
    log_factor_as_target = sampler_latents.log_factor(proposals['image_means'], priors['samples'])
    log_marginal_from_is = sampler_latents.log_marginal(proposals['image_means'].detach(), sampler_images.prior_nat1, sampler_images.prior_nat2)
    w_phi = F.softmax(log_factor_as_target - proposals['log_prob'], 0).detach()
    
    unnormalized = log_marginal_from_q.mean() 
    normalizing_constant = (w_phi * log_marginal_from_is).sum(0).mean()
    loss_phi = - (unnormalized - normalizing_constant)
    
    q_proposal = sampler_latents.forward(images, sampler_images.prior_nat1, sampler_images.prior_nat2)
    p_proposal = sampler_images.forward(images, q_proposal['samples'].detach())
    log_joint_target = sampler_latents.log_joint(p_proposal['image_means'], q_proposal['samples'], sampler_images.prior_nat1, sampler_images.prior_nat2)
    w_theta = F.softmax(log_joint_target - p_proposal['log_prob'] - q_proposal['log_prob'], 0).detach()
    loss_theta = - (w_theta * p_proposal['log_prob']).sum(0).mean()


    return loss_phi, loss_theta, unnormalized, normalizing_constant




# def log_likelihood_images(neural_ss, latents):
#     return (neural_ss * latents).sum(-1)
    
