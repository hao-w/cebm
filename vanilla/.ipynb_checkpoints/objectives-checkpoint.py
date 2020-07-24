import torch.nn.functional as F
from torch import logsumexp


def mle(ef, sgld_sampler, data_images, sgld_num_steps, sgld_step_size, buffer_size, buffer_percent):
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
    return energy_data, energy_ebm
    

