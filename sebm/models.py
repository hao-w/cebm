import math
from sebm.nets import SimpleNet, SimpleNet2, SimpleNet3, SimpleNet4, SimpleNet5, MLPNet, MLPNet2, Wide_Residual_Net, _mlp_block
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from sebm.gaussian_params import nats_to_params, params_to_nats

class Discriminator_BIGAN(nn.Module):
    """
    Discriminator in BIGAN
    """
    def __init__(self, arch, latent_dim, **kwargs):
        super().__init__()
        if arch == 'simplenet':
            self.x_enc_net = SimpleNet5(**kwargs)
            self.disc_net = SimpleNet5(im_height=1,
                                        im_width=1,
                                        input_channels=int(2*latent_dim),
                                        channels=[int(2*latent_dim), int(2*latent_dim), 1],
                                        kernels=[1,1,1],
                                        strides=[1,1,1],
                                        paddings=[0,0,0],
                                        activation=kwargs['activation'],
                                        leak=kwargs['leak'],
                                        last_act=False,
                                        batchnorm=False)
            
        else:
            raise NotImplementError # will implement wresnet-28-10 later
        self.sigmoid = nn.Sigmoid()
    
    def binary_pred(self, x, z):
        features = self.x_enc_net(x)
        assert features.shape == z.shape, 'feature shape=%s, z shape=%s' % (features.shape, z.shape)
        xz = torch.cat((features, z), dim=1)
        return self.sigmoid(self.disc_net(xz)).squeeze()
    
class Encoder_BIGAN(nn.Module):
    """
    An encoder in BIGAN
    """
    def __init__(self, arch, latent_dim, reparameterized, **kwargs):
        super(self.__class__, self).__init__()
        
        if arch == 'simplenet':
             self.enc_net = SimpleNet5(**kwargs)
        else:
            raise NotImplementError
        self.reparameterized = reparameterized
        self.arch = arch
        self.latent_dim = latent_dim
        
    def forward(self, images):
        if self.arch == 'simplenet':
            output = self.enc_net(images)
        mu, log_sigma = output[:, :self.latent_dim, :, :], output[:, self.latent_dim:, :, :]
#         print(mu.shape)
#         print(log_sigma.shape)
        q_dist = Normal(mu, log_sigma.exp())
        if self.reparameterized:
            latents = q_dist.rsample()
        else:
            latents = q_dist.sample()
        assert latents.shape == (images.shape[0], self.latent_dim, 1, 1), 'latent shape =%s' % latents.shape
        return latents
    
class Generator(nn.Module):
    """
    A Generator in GAN
    """
    def __init__(self, arch, device, **kwargs):
        super(self.__class__, self).__init__()
        if arch == 'simplenet':
            self.gen_net = SimpleNet4(**kwargs)
            self.latent_dim = kwargs['input_channels']
        else:
            raise NotImplementError
        self.tanh = nn.Tanh()
        
        self.noise_mu = torch.zeros(1).cuda().to(device)
        self.noise_sigma = torch.ones(1).cuda().to(device)
        self.noise_dist = Normal(self.noise_mu, self.noise_sigma)
        
    def forward(self, sample_size):
        z = self.noise_dist.sample((sample_size, self.latent_dim, 1, ))
        assert z.shape == (sample_size,self.latent_dim,1,1)
        x = self.tanh(self.gen_net(z))
        return z, x

class Discriminator(nn.Module):
    """
    Discriminator in a GAN
    """
    def __init__(self, arch, **kwargs):
        super().__init__()
        if arch == 'simplenet':
            self.disc_net = SimpleNet(**kwargs)
        else:
            raise NotImplementError # will implement wresnet-28-10 later
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h = self.disc_net.cnn_block(x)
        h = torch.nn.Flatten()(h)
        h = self.disc_net.mlp_block[:3](h)
        return h
    
    def binary_pred(self, x):
        return self.sigmoid(self.disc_net(x)).squeeze()
    
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
        h = self.ebm_net.cnn_block(x)
        h = torch.nn.Flatten()(h)
        h = self.ebm_net.mlp_block[:3](h)
        return h
    
    def energy(self, x):
        return self.ebm_net(x).squeeze()

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
    def __init__(self, arch, latent_dim, optimize_priors, device, **kwargs):
        super().__init__()
        if arch == 'simplenet2':
            kwargs['latent_dim'] = latent_dim
            self.ebm_net = SimpleNet2(**kwargs)
        elif arch == 'simplenet':
            self.ebm_net = SimpleNet(**kwargs)
        elif arch == 'simplenet5':
            self.ebm_net = SimpleNet5(**kwargs)
        elif arch =='wresnet':
            self.ebm_net = Wide_Residual_Net(**kwargs)
        else:
            raise NotImplementError 
            
        self.prior_nat1 = torch.zeros(latent_dim).cuda().to(device)
        self.prior_nat2 = - 0.5 * torch.ones(latent_dim).cuda().to(device)  # same prior for each pixel       
                
        if optimize_priors:
            self.prior_nat1 = nn.Parameter(self.prior_nat1)
            self.prior_nat2 = nn.Parameter(self.prior_nat2)
        self.arch = arch
        self.latent_dim = latent_dim
        
    def forward(self, x):
        if self.arch == 'simplenet2' or self.arch == 'wresnet':
            neural_ss1, neural_ss2 = self.ebm_net(x)
            return neural_ss1, - neural_ss2**2
        elif self.arch == 'simplenet':
            neural_ss1 = self.ebm_net(x)
            neural_ss2 = - neural_ss1**2
            return neural_ss1, neural_ss2
        elif self.arch == 'simplenet5':
            out = self.ebm_net(x).squeeze()
            return out[:, :self.latent_dim], - out[:, self.latent_dim:]**2
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

    def log_prior(self, latents):
        """
        return samples from prior of size S * B * latent_dim
        and log_prob of size S * B
        """
        prior_mu, prior_sigma = nats_to_params(self.prior_nat1, self.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma)       
        return prior_dist.log_prob(latents).sum(-1)  
    
    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
    def log_factor(self, x, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size  B 
        """
        neural_ss1, neural_ss2 = self.forward(x)
        B, latent_dim = neural_ss1.shape
        assert latents.shape == (B, latent_dim), 'ERROR!'
        return (neural_ss1 * latents).sum(1) + (neural_ss2 * (latents**2)).sum(1)
    
    def log_factor_expand(self, x, latents, expand_dim):
        """
        compute the log heuristic factor for the EBM
        log factor of size  B 
        """
        neural_ss1, neural_ss2 = self.forward(x)
        B, latent_dim = neural_ss1.shape
        neural_ss1 = neural_ss1.repeat(expand_dim , 1, 1)
        neural_ss2 = neural_ss2.repeat(expand_dim , 1, 1)
        return (neural_ss1 * latents).sum(2) + (neural_ss2 * (latents**2)).sum(2)
    
    
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
            self.latent_dim = kwargs['mlp_input_dim']
#             self.dec_net = MLPNet(**kwargs)
#             self.latent_dim = kwargs['input_dim']
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
        S, B, C, H ,W = images.shape
        recons = self.sigmoid(self.dec_net(latents))
        recons = recons.view(S, B, C, H, W)
        p_log_prob = Normal(self.prior_mu, self.prior_sigma).log_prob(latents.view(S, B, -1)).sum(-1)
        ll = - self.binary_cross_entropy(recons, images)
        assert ll.shape == (S, B), 'll.shape = %s' % ll.shape
        return recons, ll, p_log_prob

    def forward_expand(self, latents, images):
        train_B, C, H ,W = images.shape
        test_B = latents.shape[0]
        recons = self.sigmoid(self.dec_net(latents))
        recons = recons.unsqueeze(1).repeat(1, train_B, 1, 1, 1)
        p_log_prob = Normal(self.prior_mu, self.prior_sigma).log_prob(latents).sum(-1)
        p_log_prob = p_log_prob.unsqueeze(1).repeat(1, train_B)
        ll = - self.binary_cross_entropy(recons, images.repeat(test_B, 1, 1, 1, 1))
        assert ll.shape == (test_B, train_B), 'll.shape = %s' % ll.shape
        return ll, p_log_prob
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1).sum(-1)
    
class Encoder(nn.Module):
    """
    An encoder in VAE
    """
    def __init__(self, arch, reparameterized, **kwargs):
        super(self.__class__, self).__init__()
        
        if arch == 'simplenet2':
             self.enc_net = SimpleNet2(**kwargs)
        elif arch == 'simplenet4':
            self.enc_net = SimpleNet4(**kwargs)
        else:
            raise NotImplementError
        self.reparameterized = reparameterized
        self.arch = arch
    def forward(self, images):
        S, B, C, H, W = images.shape
        if self.arch == 'mlp':
            images = images.view(S, B, C*H*W)
        elif self.arch == 'simplenet2':
            images = images.view(S*B, C, H, W)
        mu, log_sigma = self.enc_net(images)
        q_dist = Normal(mu, log_sigma.exp())
        if self.reparameterized:
            latents = q_dist.rsample()
        else:
            latents = q_dist.sample()
        log_prob = q_dist.log_prob(latents).sum(-1).view(S, B)
        return latents, log_prob
    
class Clf(nn.Module):
    """
    a (semi-)supervised classifier
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
    
class CEBM_GMM_2ss(nn.Module):
    """
    conjugate EBM with latent variable z of a GMM prior,
    where the ebm encodes each image into two 
    neural sufficient statistics tx1 tx2
    where tx2 = - tx1^2
    """
    def __init__(self, K, arch, optimize_priors, device, **kwargs):
        super().__init__()
        if arch == 'simplenet2':
            self.ebm_net = SimpleNet2(**kwargs)
        elif arch == 'simplenet':
            self.ebm_net = SimpleNet(**kwargs)
        else:
            raise NotImplementError 
        self.prior_mu = 0.31 * torch.randn((K, kwargs['latent_dim'])).cuda().to(device)
        self.prior_log_sigma = (5*torch.rand((K, kwargs['latent_dim'])) + 1.0).log().cuda().to(device)
#         mu = 0.31 * torch.randn((K, kwargs['latent_dim']))
#         std = (5*torch.rand((K, kwargs['latent_dim'])) + 1.0)
#         self.prior_nat1 = ((mu) / (std**2)).cuda().to(device)
#         self.prior_nat2 = (- 0.5 / (std**2)).cuda().to(device)  # K * D
        if optimize_priors:
            self.prior_mu = nn.Parameter(self.prior_mu)
            self.prior_log_sigma = nn.Parameter(self.prior_log_sigma)
        self.arch = arch
        self.K = K
        self.log_K = torch.Tensor([K]).log().cuda().to(device)
        
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
        prior_nat1, prior_nat2 = params_to_nats(self.prior_mu, self.prior_log_sigma.exp())
        logA_prior = self.log_partition(prior_nat1, prior_nat2) # K * D
#         logA_prior = self.log_partition(self.prior_nat1, self.prior_nat2)
        logA_posterior = self.log_partition(prior_nat1.unsqueeze(0)+neural_ss1.unsqueeze(1), prior_nat2.unsqueeze(0)+neural_ss2.unsqueeze(1)) # B * K * D
        assert logA_prior.shape == (self.K, neural_ss1.shape[1]), 'unexpected shape.'
        assert logA_posterior.shape == (neural_ss1.shape[0], self.K, neural_ss1.shape[-1]), 'unexpected shape.'
        return self.log_K - torch.logsumexp(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)   
 
    def energy_cond(self, x, y):
        """
        """
        neural_ss1, neural_ss2 = self.forward(x)
        prior_nat1, prior_nat2 = params_to_nats(self.prior_mu, self.prior_log_sigma.exp())
        logA_prior = self.log_partition(prior_nat1, prior_nat2) # K * D
#         logA_prior = self.log_partition(self.prior_nat1, self.prior_nat2)
        logA_posterior = self.log_partition(prior_nat1.unsqueeze(0)+neural_ss1.unsqueeze(1), prior_nat2.unsqueeze(0)+neural_ss2.unsqueeze(1)) # B * K * D
        assert logA_prior.shape == (self.K, neural_ss1.shape[1]), 'unexpected shape.'
        assert logA_posterior.shape == (neural_ss1.shape[0], self.K, neural_ss1.shape[-1]), 'unexpected shape.'
        return (logA_posterior.sum(2) - logA_prior.sum(1))[:, y] 
     
        
    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
    def log_factor(self, x, latents):
        """
        compute the log heuristic factor for the EBM
        log factor of size  B 
        """
        neural_ss1, neural_ss2 = self.forward(x)
        B, latent_dim = neural_ss1.shape
        assert latents.shape == (B, latent_dim), 'ERROR!'
        return (neural_ss1 * latents).sum(1) + (neural_ss2 * (latents**2)).sum(1)
    
    def posterior_y(self, x):
        """
        p(y | x)
        """
        neural_ss1, neural_ss2 = self.forward(x)
        prior_nat1, prior_nat2 = params_to_nats(self.prior_mu, self.prior_log_sigma.exp())
        logA_prior = self.log_partition(prior_nat1, prior_nat2) # K * D
#         logA_prior = self.log_partition(self.prior_nat1, self.prior_nat2)
        logA_posterior = self.log_partition(prior_nat1.unsqueeze(0)+neural_ss1.unsqueeze(1), prior_nat2.unsqueeze(0)+neural_ss2.unsqueeze(1)) # B * K * D
        assert logA_prior.shape == (self.K, neural_ss1.shape[1]), 'unexpected shape.'
        assert logA_posterior.shape == (neural_ss1.shape[0], self.K, neural_ss1.shape[-1]), 'unexpected shape.'
        probs = torch.nn.functional.softmax(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)
        return probs