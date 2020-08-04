import torch.nn.functional as F

def mle(ef, proposal, data_images):
    """
    objective that minimizes the KL (p^{DATA} (x) || p_\theta (x)),
    or maximzie the likelihood:
    '''
    -\nabla_\theta E_{p^{DATA}(x)} [\log \frac{p^{DATA} (x)}{p_\theta (x)}]
    = \nabla_\theta E_{p^{DATA}(x)} [-E_\theta(x) - \log Z_\theta]
    = - \nabla_\theta (E_{p^{DATA}(x)} [E_\theta(x)] - E_{p_\theta(x)}[E(x)])
    '''
    we acquire samples from ebm using stochastic gradient langevin dynamics
    """ 
    batch_size, C, pixels_size, _ = data_images.shape
    energy_data = ef.forward(data_images)
    latents_samples, latents_prior_log_pdf = ef.priors(batch_size=batch_size)
    proposal_samples, proposal_log_pdf = proposal.forward(latents=latents_samples)
    proposal_samples = (proposal_samples.view(batch_size, pixels_size, pixels_size).unsqueeze(1) - 0.5) / 0.5
    energy_ebm, ll = ef.forward(proposal_samples, latents=latents_samples)
    w = F.softmax(ll - proposal_log_pdf, 0).detach()
    loss_theta = energy_data.mean() - (w * energy_ebm).sum(0)
    loss_phi = (- w * proposal_log_pdf).sum(0)
#     loss = energy_data.sum(-1).sum(-1).mean() - energy_ebm.sum(-1).sum(-1).mean()
    return loss_theta, loss_phi
    

