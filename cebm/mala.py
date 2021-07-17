import time
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from tqdm import tqdm

class MALA_Sampler():
    """
    An invalid sampler inspired by MALA
    """
    def __init__(self, im_h, im_w, im_channels, device, batch_size, latent_dim):
        super().__init__()
        self.im_dims = (im_channels, im_h, im_w)
        self.device = device
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
    def refine_MALA(self, models, num_steps, lr, z=None, eps=None):
        if z is None or eps is None:
            eps = torch.randn((self.batch_size, *self.im_dims), device=self.device)
            z = torch.randn((self.batch_size, self.latent_dim), device=self.device)
        
        vs = (z.requires_grad_(), eps.requires_grad_())
        steps = [vs]
        accepts = []
        # gfn compute the reparameterized x = eps * sigma + mu
        # efn compute the log unnormalized density of ebm, i.e. - E(x)
        gfn = lambda z, eps: models['gen'].forward(z) + models['gen'].x_logsigma.exp() * eps
        efn = lambda z, eps: - models['ebm'].energy(gfn(z, eps))
        with torch.no_grad():
            x_init = gfn(z, eps)
        for k in tqdm(range(num_steps)):
            vs, a = self.MALA(vs, efn, lr)
            steps.append(vs)
            accepts.append(a.item())
        ar = torch.tensor(accepts).mean()
        z_final, eps_final = steps[-1]
        with torch.no_grad():
            x_final = gfn(z_final, eps_final)

        return x_init, x_final

    def plot_samples(self, x_init, x_final, denormalize, fs=1):
        test_batch_size = len(images)
        x_init = x_init.squeeze().cpu().detach()
        x_init = torch.clamp(x_init, min=-1, max=1)
        if denormalize:
            x_init = x_init * 0.5 + 0.5
        gs = gridspec.GridSpec(int(test_batch_size/10), 10)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.1, hspace=0.1)
        fig = plt.figure(figsize=(fs*10, fs*int(test_batch_size/10)))
        for i in range(test_batch_size):
            ax = fig.add_subplot(gs[int(i/10), i%10])
            try:
                ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(images[i], (1,2,0)), vmin=0, vmax=1.0)
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
        
    def log_joint(self, var, mu, std):
        """
        log prob of a Nomral distribution
        """
        return Normal(mu, std).log_prob(var).flatten(start_dim=1).sum(1)
                        
    def MALA(self, vs, efn, lr):
        """
        Metropolis-Adjusted Langevin Algorithm.
        """
        step_std = (2 * lr) ** .5
        log_gamma = efn(*vs)
        grads = torch.autograd.grad(log_gamma.sum(), vs)
        updates_mu = [v + lr * g for v, g in zip(vs, grads)]
        vs_proposed = [u_mu + step_std * torch.randn_like(u_mu) for u_mu in updates_mu]
        log_gamma_proposed = efn(*vs_proposed)
        reverse_grads = torch.autograd.grad(log_gamma_proposed.sum(), vs_proposed)
        reverse_updates_mu = [v + lr * g for v, g in zip(vs_proposed, reverse_grads)]

        log_forward = sum([self.log_joint(u, u_mu, step_std) for u, u_mu in zip(vs_proposed, updates_mu)])
        log_reverse = sum([self.log_joint(v, ru_mu, step_std) for v, ru_mu in zip(vs, reverse_updates_mu)])
        logp_accept = log_gamma_proposed + log_reverse - log_gamma - log_forward
        p_accept = logp_accept.exp()
        accept = (torch.rand_like(p_accept) < p_accept).float()

        next_vars = []
        for u_v, v in zip(vs_proposed, vs):
            if len(u_v.size()) == 4:
                next_vars.append(accept[:, None, None, None] * u_v + (1 - accept[:, None, None, None]) * v)
            else:
                next_vars.append(accept[:, None] * u_v + (1 - accept[:, None]) * v)
        return next_vars, accept.mean()


    def sample(self, ebm, batch_size, num_steps, pcd=True, init_samples=None):
        """
        perform update using slgd
        pcd means that we sample from replay buffer (with a frequency)
        """
        if pcd:
            samples, inds = self.sample_from_buffer(batch_size)
        else:
            if init_samples is None:
                samples = self.initial_dist.sample((batch_size, ))
            else:
                samples = init_samples
        
        list_samples = []
        for l in range(num_steps):
            samples.requires_grad = True
            grads = torch.autograd.grad(outputs=ebm.energy(samples).sum(), inputs=samples)[0]
            samples = (samples - (self.alpha / 2) * grads + self.noise_std * torch.randn_like(grads)).detach()
            #added this extra detachment step, becase the last update keeps the variable in the graph.
            samples = samples.detach() 
        assert samples.requires_grad == False, "samples should not require gradient."
        if pcd:
            self.refine_buffer(samples.detach(), inds)
        return samples
