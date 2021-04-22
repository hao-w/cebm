import torch
import torch.nn as nn
import numpy as np
from cebm.net import build_decoder, Swish, cnn_mlp_2out, mlp_dcnn
from cebm.utils import cnn_output_shape
import torch.distributions as dists


class VAE(nn.Module):
    def __init__(self, image_height, image_width, n_color_chan, net, num_neurons=[512],
                 channels=[64,64,32,32], kernels=[4,4,4,4], strides=[2,2,2,2],
                 paddings=[3,1,1,1], activation='Swish', num_latent=128):
        super().__init__()
        self.image_height = image_height  # for easier reshaping
        self.image_width = image_width
        self.n_color_chan = n_color_chan
        self.num_pixels = image_height * image_width * n_color_chan
        self.num_latent = num_latent

        assert net in ['mlp', 'cnn']
        self.net = net
        enc_net = getattr(nets, "_"+net)
        enc, dim_last = enc_net(image_width, image_width, n_color_chan, num_neurons=num_neurons,
                                 channels=channels, kernels=kernels, strides=strides,
                                 paddings=paddings, activation=activation)

        self.enc = enc # nn.Sequential(enc1, enc2)
        self.dim_last = dim_last

        dec = build_decoder(enc)
        dec = [nn.Linear(self.num_latent, self.dim_last)] + dec + [nn.Sigmoid()]
        self.dec = nn.Sequential(*dec)

        self.z_mu = nn.Linear(self.dim_last, self.num_latent)
        self.z_logvar = nn.Linear(self.dim_last, self.num_latent)

    def encode(self, x, labels=None):
        shared = self.enc(x)
        return self.z_mu(shared), self.z_logvar(shared)

    def extract_zs(self, x):
        return self.z_mu(self.enc(x))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # size: (B, D)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def sample(self, device, n=10):
        z = torch.randn(n, self.num_latent).to(device)
        return self.decode(z)

    def forward(self, x, labels=None):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

class GMM_VAE(VAE):
    def __init__(self, image_height, image_width, n_color_chan, net, num_neurons=[512],
                 channels=[64,64,32,32], kernels=[4,4,4,4], strides=[2,2,2,2],
                 paddings=[3,1,1,1], activation='Swish', num_latent=128, K=10):
        super().__init__(image_height, image_width, n_color_chan, net, num_neurons,
                     channels, kernels, strides, paddings, activation, num_latent)
        self.K = K
        mus = nn.Parameter(torch.randn((self.K, num_latent))*0.31)
        log_stds = nn.Parameter((5*torch.rand((self.K, num_latent)) + 1.0).log())
        self.register_parameter("mus", mus)
        self.register_parameter("log_stds", log_stds)

    def prior_prob(self, z):
        log_p = 0.0
        dist_ks = dists.Normal(self.mus.unsqueeze(-2),
                               self.log_stds.unsqueeze(-2).exp())
        log_p = dist_ks.log_prob(z.unsqueeze(0)).sum(-1).logsumexp(dim=0) - np.log(self.K)
        return log_p

    def sample(self, device, n):
        n_k = int(n / self.K)
        ks = np.random.choice(self.K, n_k)
        samples = []
        for k in ks:
            mixture = dists.Normal(self.mus[k], self.log_stds[k].exp())
            samples.append(mixture.sample(torch.Size([1])))
        return self.decode(torch.cat(samples, 0))

    def forward(self, x, labels=None):
        # Concatenate the color channels into a tall vector
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), z, mu, logvar, self.prior_prob(z)

#FIXME: hao's code on vae encoder and decoder. This will be cleaned up later
class Decoder(nn.Module):
    def __init__(self, optimize_prior, device, **kwargs):
        super().__init__()
        self.dec_net = nn.Sequential(*(list(mlp_dcnn(**kwargs)) + [nn.Sigmoid()]))
        self.prior_mean = torch.zeros(kwargs['mlp_input_dim']).to(device)
        self.prior_log_std = torch.zeros(kwargs['mlp_input_dim']).to(device)
        if optimize_prior:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_log_std = nn.Parameter(self.prior_log_std)
            
    def forward(self, z, x):
        S, B, C, H ,W = x.shape
        recon = self.dec_net(z.view(S*B, -1)).view(S, B, C, H, W)
        log_pz = dists.Normal(self.prior_mean, self.prior_log_std.exp()).log_prob(z).sum(-1)
        ll = - self.binary_cross_entropy(recon, x)
        return recon, ll, log_pz
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1).sum(-1)
    
class Decoder_GMM(nn.Module):
    def __init__(self, optimize_prior, device, gmm_components, **kwargs):
        super().__init__()
        self.dec_net = nn.Sequential(*(list(mlp_dcnn(**kwargs)) + [nn.Sigmoid()]))
        self.prior_means = 0.31 * torch.randn((gmm_components, kwargs['mlp_input_dim']), device=device)
        self.prior_log_stds = (5*torch.rand((gmm_components, kwargs['mlp_input_dim']), device=device) + 1.0).log()
        if optimize_prior:
            self.prior_means = nn.Parameter(self.prior_means)
            self.prior_log_stds = nn.Parameter(self.prior_log_stds)
        self.prior_pi = torch.ones(gmm_components, device=device) / gmm_components

    def forward(self, z, x):
        S, B, C, H ,W = x.shape
        k = dists.OneHotCategorical(probs=self.prior_pi).sample((S*B,)).argmax(-1)
        prior_mean_sampled = self.prior_means[k].view(S,B,-1)
        prior_log_std_sampled = self.prior_log_stds[k].view(S,B,-1)
        recon = self.dec_net(z.view(S*B, -1)).view(S, B, C, H, W)
        log_pz = dists.Normal(prior_mean_sampled, prior_log_std_sampled.exp()).log_prob(z).sum(-1)
        ll = - self.binary_cross_entropy(recon, x)
        assert ll.shape == (S,B)
        assert log_pz.shape == (S,B)
        return recon, ll, log_pz
    
    def binary_cross_entropy(self, x_mean, x, EPS=1e-9):
        return - (torch.log(x_mean + EPS) * x + 
                  torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1).sum(-1).sum(-1)
    
class Encoder(nn.Module):
    def __init__(self, reparameterized, **kwargs):
        super().__init__()
        self.enc_cnn, self.enc_mean_net, self.enc_log_std_net = cnn_mlp_2out(**kwargs)
        self.reparameterized = reparameterized
        
    def forward(self, x):
        mean, std = self.latent_params(x)
        q = dists.Normal(mean, std)
        z = q.rsample() if self.reparameterized else q.sample()
        log_qz = q.log_prob(z).sum(-1)
        return z.view(S,B,-1), log_qz.view(S,B)
    
    def latent_params(self, x):
        S, B, C, H ,W = x.shape
        h = self.enc_cnn(x.view(S*B, C, H, W))
        return self.enc_mean_net(h), self.enc_log_std_net(h).exp()
    
def parse_decoder_network_args(im_h, im_w, im_channels, args):
    dcnn_input_h, dcnn_input_w = cnn_output_shape(im_h, 
                                                  im_w, 
                                                  eval(args.enc_kernels), 
                                                  eval(args.enc_strides), 
                                                  eval(args.enc_paddings))
    dec_network_args = {'mlp_input_dim': args.latent_dim, 
                        'hidden_dim': eval(args.hidden_dim),
                        'dcnn_input_height': dcnn_input_h,
                        'dcnn_input_width': dcnn_input_w,
                        'dcnn_input_channels': eval(args.dec_channels)[0], 
                        'channels': eval(args.dec_channels)[1:]+[im_channels], 
                        'kernels': eval(args.dec_kernels), 
                        'strides': eval(args.dec_strides), 
                        'paddings': eval(args.dec_paddings),
                        'activation': args.activation}
    return dec_network_args