import torch.nn.functional as F

def marginal_kl(ebm, proposal, data_images, sample_size, num_patches, regularize_alpha):
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
    latents, _ = ebm.sample_priors(sample_size, batch_size, num_patches)
    images_ebm, ll = proposal(latents)
    nerual_ss1_ebm = ebm.forward(images_ebm, dist='ebm')
    energy_ebm = ebm.energy(nerual_ss1_ebm, dist='ebm')
    log_factor_ebm = ebm.log_factor(nerual_ss1_ebm, latents)
    w = F.softmax(log_factor_ebm - ll, 0).detach()
    trace['ess'] = (1 / (w**2).sum(0)).mean()
    trace['loss_theta'] = (energy_data -  (w * energy_ebm).sum(0)).sum(-1).sum(-1).mean()
    trace['loss_phi'] = (w * ( - ll)).sum(0).sum(-1).sum(-1).mean()
#     loss = energy_data.sum(-1).sum(-1).mean() - energy_ebm.sum(-1).sum(-1).mean()
#     if regularize_alpha is not None:
#         regularize_term = regularize_alpha * ((energy_data**2).mean() + (energy_ebm**2).mean())
#     else:
#         regularize_term = 0
    return trace


def rws(enc, dec, images):
    """
    compute the EUBO in rws and the separate gradient w.r.t. phi and theta
    """
    trace = dict()
    latents, q_log_pdf = enc(images)
    p_log_pdf, recon, ll = dec(latents, images)
    log_w = (ll + p_log_pdf - q_log_pdf).detach()
    w = F.softmax(log_w, 0)
    trace['loss_theta'] = (- w * ll).sum(0).mean()
    trace['loss_phi'] = (- w * q_log_pdf).sum(0).mean()
    trace['eubo'] = (w * log_w).sum(0).mean()
    trace['elbo'] = log_w.mean()
    trace['ess'] = (1 / (w**2).sum(0)).mean()
    return trace 
    

def vae(enc, dec, images):
    """
    compute the ELBO in vae
    """
    trace = dict()
    latents, q_log_pdf = enc(images)
    p_log_pdf, recon, ll = dec(latents, images)
    log_w = (ll + p_log_pdf - q_log_pdf)
    trace['elbo'] = log_w.mean()
    return trace 

def ae(enc, dec, images):
    """
    compute the ELBO in vae
    """
    trace = dict()
    latents = enc(images)
    recon, ll = dec(latents, images)
    trace['loss'] = - ll.mean()
    return trace 