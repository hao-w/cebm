import torch.nn.functional as F

def mle(ef, sgld_sampler, data_images, sgld_num_steps, sgld_step_size, buffer_size, buffer_percent, regularize_alpha=None):
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
    ebm_images = sgld_sampler.sgld_update(ef=ef, 
                                          batch_size=batch_size, 
                                          pixels_size=pixels_size, 
                                          num_steps=sgld_num_steps, 
                                          step_size=sgld_step_size,
                                          buffer_size=buffer_size,
                                          buffer_percent=buffer_percent,
                                          persistent=True)
    energy_ebm = ef.forward(ebm_images)
    loss_theta = energy_data.mean() -  energy_ebm.mean()
#     loss = energy_data.sum(-1).sum(-1).mean() - energy_ebm.sum(-1).sum(-1).mean()
    if regularize_alpha is not None:
        regularize_term = regularize_alpha * ((energy_data**2).mean() + (energy_ebm**2).mean())
    else:
        regularize_term = 0
    return loss_theta, regularize_term
    

