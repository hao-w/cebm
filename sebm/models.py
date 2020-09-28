import math
from sebm.nets import SimpleNet, SimpleNet2, SimpleNet3, MLPNet, MLPNet2, Wide_Residual_Net, _mlp_block
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from sebm.gaussian_params import nats_to_params, params_to_nats
                   
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
        if arch == 'simplenet2':
            self.ebm_net = SimpleNet2(**kwargs)
        elif arch == 'simplenet':
            self.ebm_net = SimpleNet(**kwargs)
        elif arch =='wresnet':
            self.ebm_net = Wide_Residual_Net(**kwargs)
        else:
            raise NotImplementError 
            
        self.prior_nat1 = torch.zeros(kwargs['latent_dim']).cuda().to(device)
        self.prior_nat2 = - 0.5 * torch.ones(kwargs['latent_dim']).cuda().to(device)  # same prior for each pixel       
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
        self.arch = arch
            
    def forward(self, x):
        if self.arch == 'simplenet2' or self.arch == 'wresnet':
            neural_ss1, neural_ss2 = self.ebm_net(x)
            return neural_ss1, - neural_ss2**2
        elif self.arch == 'simplenet':
            neural_ss1 = self.ebm_net(x)
            neural_ss2 = - neural_ss1**2
            return neural_ss1, neural_ss2
        else:
            raise NotImplementError
    
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
    
    def sample_prior(self, sample_size, batch_size):
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
    
    
    def sample_posterior(self, sample_size, neural_ss1, neural_ss2):
        """
        return samples from the conjugate posterior
        """
        posterior_mu, posterior_sigma = nats_to_params(self.prior_nat1+neural_ss1, self.prior_nat2+neural_ss2)
        posterior_dist = Normal(posterior_mu, posterior_sigma) 
        latents = posterior_dist.sample((sample_size, ))
        return latents, posterior_dist.log_prob(latents).sum(-1)

class Decoder(nn.Module):
    """
    A decoder in VAE
    """
    def __init__(self, arch, device, **kwargs):
        super(self.__class__, self).__init__()
        if arch == 'simplenet2':
            self.dec_net = SimpleNet3(**kwargs)
            self.latent_dim = kwargs['latent_dim']
        elif arch == 'mlp':
            self.dec_net = MLPNet(**kwargs)
            self.latent_dim = kwargs['input_dim']
        else:
            raise NotImplementError
        self.sigmoid = nn.Sigmoid()
        self.arch = arch
        
        self.prior_mu = torch.zeros(self.latent_dim).cuda().to(device)
        self.prior_sigma = torch.ones(self.latent_dim).cuda().to(device)
        
    def forward(self, latents, images):
        recons = self.sigmoid(self.dec_net(latents))
        assert recons.shape == images.shape, 'shape of recons : %s, shape of images : %s' % (recons.shape, images.shape)
        if self.arch == 'simplenet2':
            SB, in_c, im_h, im_w = images.shape
            recons = recons.view(SB, in_c, im_h*im_h)
            images = images.contiguous().view(SB, in_c, im_h*im_h)
        else:
            raise NotImplementError
        p_log_prob = Normal(self.prior_mu, self.prior_sigma).log_prob(latents).sum(-1)
        ll = - self.binary_cross_entropy(recons, images)
        return recons.view(SB, in_c, im_h, im_h), ll, p_log_prob
            
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1)
    
class Encoder(nn.Module):
    """
    An encoder in VAE
    """
    def __init__(self, arch, reparameterized, **kwargs):
        super(self.__class__, self).__init__()
        
        if arch == 'simplenet2':
             self.enc_net = SimpleNet2(**kwargs)
        elif arch == 'mlp':
            self.enc_net = MLPNet2(**kwargs)
        else:
            raise NotImplementError
        self.reparameterized = reparameterized
        
    def forward(self, images):
        mu, log_sigma = self.enc_net(images)
        q_dist = Normal(mu, log_sigma.exp())
        if self.reparameterized:
            latents = q_dist.rsample().squeeze()
        else:
            latents = q_dist.sample()
        log_prob = q_dist.log_prob(latents).sum(-1)
        return latents, log_prob
    
class Clf(nn.Module):
    """
    a naive supervised classifier
    """
    def __init__(self, arch, **kwargs):
        super().__init__()
        if arch =='simplenet':
            self.hidden = SimpleNet(**kwargs)
        elif arch == 'mlp':
            self.hidden = _mlp_block(**kwargs)
        else:
            raise NotImplementError
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss()
        
    def forward(self, x):
        """
        return log probability
        """
        return self.logsoftmax(self.hidden(x))
    
    def loss(self, pred_y, y):
        return self.nllloss(pred_y, y)
    
    def score(self, pred_y, y):
        return (pred_y.argmax(-1) == y).float().sum()