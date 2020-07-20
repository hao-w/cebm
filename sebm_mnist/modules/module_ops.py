import torch
from quasi_conj.modules.sampler_latents import Sampler_latents
from quasi_conj.modules.sampler_images import Sampler_images

def init_modules(pixels_dim, hidden_dim, latents_dim, optimize_priors, reparameterized, CUDA, DEVICE, LOAD_VERSION=None, LR=None):
    """
    initialize modules
    """
    sampler_latents = Sampler_latents(pixels_dim=pixels_dim, hidden_dim=hidden_dim, neural_ss_dim=latents_dim, optimize_priors=optimize_priors, reparameterized=reparameterized)
    sampler_images = Sampler_images(latents_dim=latents_dim, hidden_dim=hidden_dim, pixels_dim=pixels_dim, reparameterized=reparameterized, CUDA=CUDA, DEVICE=DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            sampler_latents.cuda()
            sampler_images.cuda()
    if LOAD_VERSION is not None:
        sampler_latents.load_state_dict(torch.load('../weights/sampler-latents-%s' % LOAD_VERSION))
        sampler_images.load_state_dict(torch.load('../weights/sampler-images-%s' % LOAD_VERSION))
    if LR is not None:
        assert isinstance(LR, float)
        optimizer_phi =  torch.optim.Adam(list(sampler_latents.parameters()),lr=LR, betas=(0.9, 0.99))   
        optimizer_theta = torch.optim.Adam(list(sampler_images.parameters()), lr=LR, betas=(0.9, 0.99))
#         optimizer = torch.optim.Adam(list(sampler_latents.parameters())+list(sampler_images.parameters()),lr=LR, betas=(0.9, 0.99))
        return (sampler_latents, sampler_images), optimizer_phi, optimizer_theta
    else:
        for p in sampler_latents.parameters():
            p.requires_grad = False
        for p in sampler_images.parameters():
            p.requires_grad = False    
        return (sampler_latents, sampler_images)
    
def save_modules(modules, SAVE_VERSION):
    """
    ==========
    saving function
    ==========
    """
    (sampler_latents, sampler_images) = modules
    torch.save(sampler_latents.state_dict(), "../weights/sampler-latents-%s" % SAVE_VERSION)
    torch.save(sampler_images.state_dict(), "../weights/sampler-images-%s" % SAVE_VERSION)
