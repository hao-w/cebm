import torch.nn.functional as F

def marginal_kl_multilayers(ebms, proposals, data_images, sample_size, num_patches, reg_alpha):
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
    (ebm1, ebm2, ebm3) = ebms
    (proposal1, proposal2, proposal3) = proposals
    (p1, p2, p3) = num_patches
    # first layer
    batch_size, C, pixels_size, _ = data_images.shape
    neural_ss1_data1 = ebm1.forward(data_images, dist='data')
    energy_data1 = ebm1.energy(neural_ss1_data1, dist='data')
    # compute the expectation w.r.t. ebm distribut5tion
    latents1, _ = ebm1.sample_priors(sample_size, batch_size, num_patches=p1)
    images_ebm1, ll1 = proposal1(latents1)
    nerual_ss1_ebm1 = ebm1.forward(images_ebm1, dist='ebm')
    energy_ebm1 = ebm1.energy(nerual_ss1_ebm1, dist='ebm')
    log_factor_ebm1 = ebm1.log_factor(nerual_ss1_ebm1, latents1)
    w1 = F.softmax(log_factor_ebm1 - ll1, 0).detach()
    trace['ess1'] = (1 / (w1**2).sum(0)).mean()
    trace['loss_theta1'] = (energy_data1 -  (w1 * energy_ebm1).sum(0)).sum(-1).sum(-1).mean()
    trace['loss_phi1'] = (w1 * ( - ll1)).sum(0).sum(-1).sum(-1).mean()
    trace['energy_data1'] = energy_data1.sum(-1).sum(-1).mean().detach()
    trace['energy_ebm1'] = (w1 * energy_ebm1).sum(0).sum(-1).sum(-1).mean().detach()
    if reg_alpha != 0.0:
        trace['regularize_term1'] = reg_alpha * ((energy_data1**2).sum(-1).sum(-1).mean() + (energy_ebm1**2).sum(-1).sum(-1).mean())
        
    # second layer
    batch_size, C, pixels_size, _ = neural_ss1_data1.shape
    neural_ss1_data2 = ebm2.forward(neural_ss1_data1, dist='data')
    energy_data2 = ebm2.energy(neural_ss1_data2, dist='data')
    # compute the expectation w.r.t. ebm distribut5tion
    latents2, _ = ebm2.sample_priors(sample_size, batch_size, num_patches=p2)
    images_ebm2, ll2 = proposal2(latents2)
    nerual_ss1_ebm2 = ebm2.forward(images_ebm2, dist='ebm')
    energy_ebm2 = ebm2.energy(nerual_ss1_ebm2, dist='ebm')
    log_factor_ebm2 = ebm2.log_factor(nerual_ss1_ebm2, latents2)
    w2 = F.softmax(log_factor_ebm2 - ll2, 0).detach()
    trace['ess2'] = (1 / (w2**2).sum(0)).mean()
    trace['loss_theta2'] = (energy_data2 -  (w2 * energy_ebm2).sum(0)).sum(-1).sum(-1).mean()
    trace['loss_phi2'] = (w2 * ( - ll2)).sum(0).sum(-1).sum(-1).mean()
    trace['energy_data2'] = energy_data2.sum(-1).sum(-1).mean().detach()
    trace['energy_ebm2'] = (w2 * energy_ebm2).sum(0).sum(-1).sum(-1).mean().detach()
    if reg_alpha != 0.0:
        trace['regularize_term2'] = reg_alpha * ((energy_data2**2).sum(-1).sum(-1).mean() + (energy_ebm2**2).sum(-1).sum(-1).mean())
        
    # third layer
    batch_size, C, pixels_size, _ = neural_ss1_data2.shape
    neural_ss1_data3 = ebm3.forward(neural_ss1_data2, dist='data')
    energy_data3 = ebm3.energy(neural_ss1_data3, dist='data')
    # compute the expectation w.r.t. ebm distribut5tion
    latents3, _ = ebm3.sample_priors(sample_size, batch_size, num_patches=p3)
    images_ebm3, ll3 = proposal3(latents3)
    nerual_ss1_ebm3 = ebm3.forward(images_ebm3, dist='ebm')
    energy_ebm3 = ebm3.energy(nerual_ss1_ebm3, dist='ebm')
    log_factor_ebm3 = ebm3.log_factor(nerual_ss1_ebm3, latents3)
    w3 = F.softmax(log_factor_ebm3 - ll3, 0).detach()
    trace['ess3'] = (1 / (w3**2).sum(0)).mean()
    trace['loss_theta3'] = (energy_data3 -  (w3 * energy_ebm3).sum(0)).sum(-1).sum(-1).mean()
    trace['loss_phi3'] = (w3 * ( - ll3)).sum(0).sum(-1).sum(-1).mean()
    trace['energy_data3'] = energy_data3.sum(-1).sum(-1).mean().detach()
    trace['energy_ebm3'] = (w3 * energy_ebm3).sum(0).sum(-1).sum(-1).mean().detach()
    if reg_alpha != 0.0:
        trace['regularize_term3'] = reg_alpha * ((energy_data3**2).sum(-1).sum(-1).mean() + (energy_ebm3**2).sum(-1).sum(-1).mean())
    return trace

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


def mle(ef, sgld_sampler, data_images, sgld_num_steps, sgld_step_size, buffer_size, buffer_percent, reg_alpha):
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
    trace = dict()
    # compute the expectation w.r.t. data distribution
    batch_size, C, pixels_size, _ = data_images.shape
    neural_ss1_data = ebm.forward(data_images, dist='data')
    energy_data = ebm.energy(neural_ss1_data, dist='data')

    images_ebm = sgld_sampler.sgld_update(ef=ef, 
                                          batch_size=batch_size, 
                                          pixels_size=pixels_size, 
                                          num_steps=sgld_num_steps, 
                                          step_size=sgld_step_size,
                                          buffer_size=buffer_size,
                                          buffer_percent=buffer_percent,
                                          persistent=True)
    
    nerual_ss1_ebm = ebm.forward(images_ebm, dist='ebm')
    energy_ebm = ebm.energy(nerual_ss1_ebm, dist='ebm')
    trace['loss_theta'] = (energy_data - energy_ebm).sum(-1).sum(-1).mean()
    trace['energy_data'] = energy_data.sum(-1).sum(-1).mean().detach()
    trace['energy_ebm'] = (w * energy_ebm).sum(0).sum(-1).sum(-1).mean().detach()
    if reg_alpha != 0.0:
        trace['regularize_term'] = reg_alpha * ((energy_data**2).sum(-1).sum(-1).mean() + (energy_ebm**2).sum(-1).sum(-1).mean())
    return trace

    energy_ebm = ef.forward(ebm_images)
    return energy_data, energy_ebm
    