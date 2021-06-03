import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
from cebm.net import cnn_block, deconv_block, mlp_block, cnn_output_shape, Swish, Reshape
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat

# class CEBM(nn.Module):
#     def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
#         super().__init__()
#         self.device = device
#         if activation == 'Swish':
#             act = Swish()
#         elif activation == 'LeakyReLU':
#             act = nn.LeakyReLU(inplace=True)
#         else:
#             act = getattr(nn, activation)()
            
#         self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1, bias=True)
#         self.res1 = BasicBlock(in_c=128, out_c=128, stride=2)
# #         self.res2 = BasicBlock(in_c=128, out_c=128, stride=1)
# #         self.res3 = BasicBlock(in_c=128, out_c=128, stride=1)
#         self.res4 = BasicBlock(in_c=128, out_c=256, stride=2)
# #         self.res5 = BasicBlock(in_c=256, out_c=256, stride=1)
# #         self.res6 = BasicBlock(in_c=256, out_c=256, stride=1)
#         self.res7 = BasicBlock(in_c=256, out_c=512, stride=2)
# #         self.res8 = BasicBlock(in_c=512, out_c=512, stride=1)
# #         self.res9 = BasicBlock(in_c=512, out_c=512, stride=1)
#         self.fc1 = nn.Sequential(
#                     nn.Linear(512*4*4, 512),
#                     act)
        
#         self.nss1_net = nn.Linear(512, latent_dim)
#         self.nss2_net = nn.Linear(512, latent_dim)
                    
#         self.flatten = nn.Flatten()
            
        
#     def forward(self, x):
#         h0 = self.conv1(x)
#         h1 = self.res1(h0)
# #         h2 = self.res2(h1)
# #         h3 = self.res3(h2)
#         h4 = self.res4(h1)
# #         h5 = self.res5(h4)
# #         h6 = self.res6(h5)
#         h7 = self.res7(h4)
# #         h8 = self.res8(h7)
# #         h9 = self.res9(h8)
#         h10 = self.fc1(self.flatten(h7))
#         nss1 = self.nss1_net(h10)
#         nss2 = self.nss2_net(h10)
#         return nss1, -nss2**2
        
#     def log_partition(self, nat1, nat2):
#         """
#         compute the log partition of a normal distribution
#         """
#         return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
#     def nats_to_params(self, nat1, nat2):
#         """
#         convert a Gaussian natural parameters its distritbuion parameters,
#         mu = - 0.5 *  (nat1 / nat2), 
#         sigma = (- 0.5 / nat2).sqrt()
#         nat1 : natural parameter which correspond to x,
#         nat2 : natural parameter which correspond to x^2.      
#         """
#         mu = - 0.5 * nat1 / nat2
#         sigma = (- 0.5 / nat2).sqrt()
#         return mu, sigma

#     def params_to_nats(self, mu, sigma):
#         """
#         convert a Gaussian distribution parameters to the natrual parameters
#         nat1 = mean / sigma**2, 
#         nat2 = - 1 / (2 * sigma**2)
#         nat1 : natural parameter which correspond to x,
#         nat2 : natural parameter which correspond to x^2.
#         """
#         nat1 = mu / (sigma**2)
#         nat2 = - 0.5 / (sigma**2)
#         return nat1, nat2    
    
#     def log_factor(self, x, latents, expand_dim=None):
#         """
#         compute the log factor log p(x | z) for the CEBM
#         """
#         nss1, nss2 = self.forward(x)
#         if expand_dim is not None:
#             nss1 = nss1.repeat(expand_dim , 1, 1)
#             nss2 = nss2.repeat(expand_dim , 1, 1)
#             return (nss1 * latents).sum(2) + (nss2 * (latents**2)).sum(2)
#         else:
#             return (nss1 * latents).sum(1) + (nss2 * (latents**2)).sum(1) 
    
#     def energy(self, x):
#         pass
    
#     def latent_params(self, x):
#         pass
        
#     def log_prior(self, latents):
#         pass   
    
        
class BasicBlock(nn.Module):
    """
    basic block module
        stride   -- stride of the 1st cnn in the 1st block in a group
        bn_flag -- whether do batch normalization
    """
    def __init__(self, in_c, out_c, stride, activation=nn.LeakyReLU(inplace=True)):
        super(BasicBlock, self).__init__()

        self.activation = activation
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, bias=True)
                    
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True)
        else:
            if stride != 1:
                self.shortcut = nn.AvgPool2d(kernel_size=2)
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        h1 = self.activation(self.c1(x))
        h2 = self.activation(self.c2(h1))
        out = self.activation(h2 + self.shortcut(x))
        
        return out  
    
class CEBM(nn.Module):
    """
    A generic class of CEBM 
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__()
        self.device = device
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
#         self.nss1_net = mlp_block(cnn_output_dim, hidden_dims, latent_dim, activation, last_act=False, **kwargs)
#         self.nss2_net = mlp_block(cnn_output_dim, hidden_dims, latent_dim, activation, last_act=False, **kwargs)
        self.mlp_net = mlp_block(cnn_output_dim, hidden_dims, activation, **kwargs)
        self.nss1_net = nn.Linear(hidden_dims[-1], latent_dim)
        self.nss2_net = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x):
        h = self.mlp_net(self.flatten(self.conv_net(x)))
        nss1 = self.nss1_net(h) 
        nss2 = self.nss2_net(h)
        return nss1, -nss2**2

    def log_partition(self, nat1, nat2):
        """
        compute the log partition of a normal distribution
        """
        return - 0.25 * (nat1 ** 2) / nat2 - 0.5 * (-2 * nat2).log()  
    
    def nats_to_params(self, nat1, nat2):
        """
        convert a Gaussian natural parameters its distritbuion parameters,
        mu = - 0.5 *  (nat1 / nat2), 
        sigma = (- 0.5 / nat2).sqrt()
        nat1 : natural parameter which correspond to x,
        nat2 : natural parameter which correspond to x^2.      
        """
        mu = - 0.5 * nat1 / nat2
        sigma = (- 0.5 / nat2).sqrt()
        return mu, sigma

    def params_to_nats(self, mu, sigma):
        """
        convert a Gaussian distribution parameters to the natrual parameters
        nat1 = mean / sigma**2, 
        nat2 = - 1 / (2 * sigma**2)
        nat1 : natural parameter which correspond to x,
        nat2 : natural parameter which correspond to x^2.
        """
        nat1 = mu / (sigma**2)
        nat2 = - 0.5 / (sigma**2)
        return nat1, nat2    
    
    def log_factor(self, x, latents, expand_dim=None):
        """
        compute the log factor log p(x | z) for the CEBM
        """
        nss1, nss2 = self.forward(x)
        if expand_dim is not None:
            nss1 = nss1.expand(expand_dim , -1, -1)
            nss2 = nss2.expand(expand_dim , -1, -1)
            return (nss1 * latents).sum(2) + (nss2 * (latents**2)).sum(2)
        else:
            return (nss1 * latents).sum(1) + (nss2 * (latents**2)).sum(1) 
    
    def energy(self, x):
        pass
    
    def latent_params(self, x):
        pass
        
    def log_prior(self, latents):
        pass   
    
    
class CEBM_Gaussian(CEBM):
    """
    conjugate EBM with a spherical Gaussian inductive bias
    """
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs)
        self.ib_mean = torch.zeros(latent_dim, device=self.device)
        self.ib_log_std = torch.zeros(latent_dim, device=self.device)
    
    def energy(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_mean, self.ib_log_std.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2)
        logA_posterior = self.log_partition(ib_nat1+nss1, ib_nat2+nss2)
        return logA_prior.sum(0) - logA_posterior.sum(1)   
    
    def latent_params(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_mean, self.ib_log_std.exp()) 
        return self.nats_to_params(ib_nat1+nss1, ib_nat2+nss2) 
    
    def log_prior(self, latents):
        return Normal(self.ib_mean, self.ib_log_std.exp()).log_prob(latents).sum(-1)      
    
class CEBM_GMM(CEBM):
    """
    conjugate EBM with a GMM inductive bias
    """
    def __init__(self, optimize_ib, num_clusters, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs)
        #Suggested initialization
        self.ib_means = 0.31 * torch.randn((num_clusters, latent_dim), device=self.device)
        self.ib_log_stds = (5*torch.rand((num_clusters, latent_dim), device=self.device) + 1.0).log()
        if optimize_ib:
            self.ib_means = nn.Parameter(self.ib_means)
            self.ib_log_stds = nn.Parameter(self.ib_log_stds)
        self.K = num_clusters
        self.log_K = torch.tensor([self.K], device=self.device).log()
    
    def energy(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_means, self.ib_log_stds.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2) # K * D
        #FIXME: Currently we only predict the same neural sufficient statistics for all components.
        logA_posterior = self.log_partition(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1)) # B * K * D
        assert logA_prior.shape == (self.K, nss1.shape[1]), 'unexpected shape.'
        assert logA_posterior.shape == (nss1.shape[0], self.K, nss1.shape[-1]), 'unexpected shape.'
        return self.log_K - torch.logsumexp(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)   
     
    def latent_params(self, x):
        nss1, nss2 = self.forward(x)
        ib_nat1, ib_nat2 = self.params_to_nats(self.ib_means, self.ib_log_stds.exp())
        logA_prior = self.log_partition(ib_nat1, ib_nat2) # K * D
        logA_posterior = self.log_partition(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1)) # B * K * D
        probs = torch.nn.functional.softmax(logA_posterior.sum(2) - logA_prior.sum(1), dim=-1)
        means, stds = nats_to_params(ib_nat1.unsqueeze(0)+nss1.unsqueeze(1), ib_nat2.unsqueeze(0)+nss2.unsqueeze(1))
        pred_y_expand = probs.argmax(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, means.shape[2])
        return torch.gather(means, 1, pred_y_expand).squeeze(1), torch.gather(stds, 1, pred_y_expand).squeeze(1)

class IGEBM(nn.Module):
    def __init__(self, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, **kwargs):
        super().__init__()
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=False, **kwargs)
        self.flatten = nn.Flatten()
        out_h, out_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        cnn_output_dim = out_h * out_w * channels[-1]
        hidden_dims.append(latent_dim)
        self.latent_net = mlp_block(cnn_output_dim, hidden_dims, activation, **kwargs)
        self.energy_net = nn.Linear(latent_dim, 1)

    def energy(self, x):
        h = self.flatten(self.conv_net(x))
        return self.energy_net(self.latent_net(h))
    
    def latent(self, x):
        return self.latent_net[:-2](self.flatten(self.conv_net(x)))

class Generator(nn.Module):
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs):
        super().__init__()
        self.device = device
        deconv_input_h, deconv_input_w = cnn_output_shape(im_height, im_width, kernels, strides, paddings)
        channels.reverse()
        kernels.reverse()
        strides.reverse()
        deconv_input_channels = channels[0]
        channels = channels[1:] + [input_channels]
        mlp_output_dim = deconv_input_channels * deconv_input_h * deconv_input_w
        hidden_dims.reverse()
        hidden_dims.append(mlp_output_dim)
        self.mlp = mlp_block(latent_dim, hidden_dims, activation, **kwargs)
        self.reshape = Reshape([deconv_input_channels, deconv_input_h, deconv_input_w])
        self.deconv_net = deconv_block(deconv_input_h, deconv_input_w, deconv_input_channels, channels, kernels, strides, dec_paddings, activation, last_act=False, batchnorm=False, **kwargs)
        self.deconv_net = nn.Sequential(*(list(self.deconv_net) + [nn.Sigmoid()]))

    def forward(self, z, x):
        pass

class Generator_VERA_Gaussian(Generator):
    def __init__(self, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs):
        super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs)
        self.prior_mean = torch.zeros(latent_dim, device=self.device)
        self.prior_log_std = torch.zeros(latent_dim, device=self.device)
        self.x_logsigma = nn.Parameter((torch.ones(1, device=self.device) * .01).log())
        
    def sample(self, batch_size):
        z0 = Normal(self.prior_mean, self.prior_log_std.exp()).sample((batch_size,))
        xr_mu = self.deconv_net(self.reshape(self.mlp(z0)))
        xr = xr_mu + torch.randn_like(xr_mu) * self.x_logsigma.exp()
        return z0, xr, xr_mu
    
    def log_joint(self, x, z):
        log_p_z = Normal(self.prior_mean, self.prior_log_std.exp()).log_prob(z).sum(-1)
        if z.dim() == 2:
            x_mu = self.deconv_net(self.reshape(self.mlp(z)))
            ll =  Normal(x_mu, self.x_logsigma.exp()).log_prob(x).sum(-1).sum(-1).sum(-1)
        elif z.dim() == 3:
            S, B, D = z.shape
            x_mu = self.deconv_net(self.reshape(self.mlp(z.view(S*B, D))))
            x_mu = x_mu.view(S, B, *x_mu.shape[1:])
            ll =  Normal(x_mu, self.x_logsigma.exp()).log_prob(x[None]).sum(-1).sum(-1).sum(-1)
        assert ll.shape == log_p_z.shape
        return ll + log_p_z, x_mu
    
class Xee(nn.Module):
    def __init__(self, device, latent_dim, init_sigma):
        super().__init__()
        self.xee_logsigma = nn.Parameter((torch.ones(latent_dim, device=device) * init_sigma).log())
        
    def sample(self, z0, sample_size=None, detach_sigma=False, entropy=False):
        if detach_sigma:
            xee_dist = Normal(z0, self.xee_logsigma.exp().detach())
        else:
            xee_dist = Normal(z0, self.xee_logsigma.exp())
            
        if sample_size is None:
            z = xee_dist.rsample()
        else:
            z = xee_dist.rsample((sample_size,))
            
        if entropy:
            ent = xee_dist.entropy().sum(-1)
            return z, ent
        else:
            lp = xee_dist.log_prob(z).sum(-1)
            return z, lp
        
# class Generator_VERA_GMM(Generator):
#     def __init__(self, optimize_prior, num_clusters, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs):
#         super().__init__(device, im_height, im_width, input_channels, channels, kernels, strides, paddings, hidden_dims, latent_dim, activation, dec_paddings, **kwargs)
#         self.prior_means = 0.31 * torch.randn((num_clusters, latent_dim), device=self.device)
#         self.prior_log_stds = (5*torch.rand((num_clusters, latent_dim), device=self.device) + 1.0).log()
#         if optimize_prior:
#             self.prior_means = nn.Parameter(self.prior_means)
#             self.prior_log_stds = nn.Parameter(self.prior_log_stds)
#         self.prior_pi = torch.ones(num_clusters, device=device) / num_clusters

#     def forward(self, z, x):
#         S, B, C, H ,W = x.shape
#         k = dists.OneHotCategorical(probs=self.prior_pi).sample((S*B,)).argmax(-1)
#         prior_mean_sampled = self.prior_means[k].view(S,B,-1)
#         prior_log_std_sampled = self.prior_log_stds[k].view(S,B,-1)
#         recon = self.dec_net(z.view(S*B, -1)).view(S, B, C, H, W)
#         log_pz = dists.Normal(prior_mean_sampled, prior_log_std_sampled.exp()).log_prob(z).sum(-1)
#         ll = - self.binary_cross_entropy(recon, x)
#         return recon, ll, log_pz    
####################################################################
class META_CEBM_Omniglot(nn.Module):
    def __init__(self, num_clusters, device, im_height, im_width, input_channels, channels, kernels, strides, paddings, latent_dim, activation):
        super().__init__()
        self.K = num_clusters
        self.device = device
        self.conv_net = cnn_block(im_height, im_width, input_channels, channels, kernels, strides, paddings, activation, last_act=True, batchnorm=True, maxpool_kernels=[2,2,2,2], maxpool_strides=[2,2,2,1])
        self.flatten = nn.Flatten()
        self.tr_enc = nn.TransformerEncoder(TransformerEncoderLayer(d_model=256, 
                                                                    nhead=4, 
                                                                    dim_feedforward=256, 
                                                                    activation='ELU'),
                                            num_layers=2)
        
        self.nss_net = nn.Linear(256, latent_dim)
#         self.nss2_net = nn.Linear(256, latent_dim)
        
        self.prior_pi = torch.ones(self.K, device=device) / self.K
        self.prior_mu = torch.zeros((self.K, latent_dim),device=device)
        self.prior_nu = torch.ones((self.K, latent_dim),device=device) * 0.5
        self.prior_alpha = torch.ones((self.K, latent_dim),device=device) * 1
        self.prior_beta = torch.ones((self.K, latent_dim),device=device) * 1
        self.log_partition_prior = self.log_partition_normal_gamma(self.prior_alpha, self.prior_beta, self.prior_nu)
        
    def forward(self, x):
        h1 = self.flatten(self.conv_net(x))
        h2 = self.tr_enc(h1.unsqueeze(0))
        return self.nss_net(h2).squeeze(0)
    
    def energy(self, x, c_means, c_stds, ys):
        """
        Option 3 based on Hao's derivation
        """
        z = self.forward(x)
        return - (self.log_factor(z.unsqueeze(1), c_means, c_stds) * ys).sum(-1)

            
    def gibbs_updates(self, x, sweeps, S):
        z = self.forward(x)
        # NOTE: precision = 1 / variance
        # initialize parameters
        c_means = torch.randn((self.K, z.shape[-1]), device=self.device) + z.mean(0).detach()
        c_stds = torch.ones((self.K, z.shape[-1]), device=self.device) # K * D
        pi = self.prior_pi # K
        for g in range(sweeps):
            gammas = self.E_step(z, c_means, c_stds, pi)
            ys = gammas # NOTE: probably need to sample instead of using the probs
            alpha, beta, mu, nu, pi = self.M_step(z, gammas)
            T = alpha / beta
            c_stds = (1 / (nu * T)).sqrt()
            c_means = mu
        return alpha, beta, mu, nu, pi, gammas, c_means.detach(), c_stds.detach(), ys.detach()    
        
    def E_step(self, z, c_means, c_stds, pi):
        lfs = self.log_factor(z.unsqueeze(1), c_means, c_stds)        
        return torch.nn.functional.softmax(pi.log() + lfs, dim=-1)
            
    def M_step(self, z, gammas):
        s1, s2, s3 = self.ss(z, gammas)
        pi = s1.squeeze(-1) / s1.sum()
        alpha = self.prior_alpha + s1 / 2
        beta = self.prior_beta + (s3 - (s2 ** 2) / s1) / 2. \
                + (s1 * self.prior_nu / (s1 + self.prior_nu)) * (((s2 / s1) - self.prior_nu)**2) / 2.
        mu = (self.prior_mu * self.prior_nu + s2) / (s1 + self.prior_nu)
        nu = self.prior_nu + s1

        return alpha, beta, mu, nu, pi
        
    def ss(self, z, gammas):
        """
        pointwise sufficient statstics
        s1 : sum of I[z_n=k], K * 1
        s2 : sum of I[z_n=k]*x_n, K * D
        s3 : sum of I[z_n=k]*x_n^2, K * D
        """
        s1 = gammas.sum(0).unsqueeze(-1)
        s1[s1 == 0.0] = 1.0
        s2 = (gammas.unsqueeze(-1) * z.unsqueeze(1)).sum(0)
        s3 = (gammas.unsqueeze(-1) * (z**2).unsqueeze(1)).sum(0)
        return s1, s2, s3
    
    def log_factor(self, z, c_means, c_stds):
        nat1, nat2 = self.params_to_nats(c_means, c_stds)
        return ((z * nat1) + (z**2) * nat2).sum(-1)
        
    def nats_to_params(self, nat1, nat2):
        return - 0.5 * nat1 / nat2, (- 0.5 / nat2).sqrt()

    def params_to_nats(self, mean, std):
        return mean / (std**2), - 0.5 / (std**2)
        
    def log_x_cond_y(self, alpha, beta, mu, nu):
        return (log_partition_normal_gamma(alpha, beta, nu) - self.log_partition_prior).sum()
        
    def log_x_cond_c(self, z, c_means, c_stds, pi):
        return self.log_partition_cat(z, c_means, c_stds, pi)
    
    def log_partition_normal_gamma(self, alpha, beta, nu):
        return torch.lgamma(alpha) - alpha * beta.log() - (1/2) * nu.log()
    
    def log_partition_cat(self, z, c_means, c_stds, pi):
        return torch.logsumexp(pi.log() + self.log_factor(z.unsqueeze(1).expand(-1, self.K, -1), c_means, c_stds), dim=-1)
    
    
class TransformerEncoderLayer(nn.Module):
    """
    Override the pytorch code by allowing ELU activation and remove the LayerNorm
    
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation="elu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = getattr(nn, activation)()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src