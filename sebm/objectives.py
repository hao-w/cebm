import torch.nn.functional as F

def marginal_kl_1layer(ebm, proposal, data_images, sample_size, reg_alpha):
    """
    objective that minimizes the KL (p^{DATA} (x) || p_\theta (x)),
    or maximzie the likelihood:
    '''
    -\nabla_\theta E_{p^{DATA}(x)} [\log \frac{p^{DATA} (x)}{p_\theta (x)}]
    = \nabla_\theta E_{p^{DATA}(x)} [-E_\theta(x) - \log Z_\theta]
    = - \nabla_\theta (E_{p^{DATA}(x)} [E_\theta(x)] - E_{p_\theta(x)}[E(x)])
    '''
    we train a proposal to get samples from ebm
    """ 
    trace = dict()
    # compute the expectation w.r.t. data distribution
    batch_size, C, pixels_size, _ = data_images.shape
    neural_ss1_data = ebm.forward(data_images, dist='data')
    energy_data = ebm.energy(neural_ss1_data, dist='data')
    # compute the expectation w.r.t. ebm distribution
    latents, _ = ebm.sample_priors(sample_size, batch_size)
    images_ebm, ll = proposal(latents)
    nerual_ss1_ebm = ebm.forward(images_ebm, dist='ebm')
    energy_ebm = ebm.energy(nerual_ss1_ebm, dist='ebm')
    log_factor_ebm = ebm.log_factor(nerual_ss1_ebm, latents)
    w = F.softmax(log_factor_ebm - ll, 0).detach()
    trace['ess'] = (1 / (w**2).sum(0)).mean()
    trace['loss_theta'] = (energy_data -  (w * energy_ebm).sum(0)).mean()
    trace['loss_phi'] = (w * ( - ll)).sum(0).mean()
    trace['energy_data'] = energy_data.mean().detach()
    trace['energy_ebm'] = (w * energy_ebm).sum(0).mean().detach()
    if reg_alpha != 0.0:
        trace['regularize_term'] = reg_alpha * ((energy_data**2).mean() + (energy_ebm**2).mean())
    return trace
    