import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ffebm.data import load_mnist
from ffebm.data_noise import DATA_NOISE_sampler
from ffebm.nets.ffebm_onelayer import Energy_function
from ffebm.nets.proposal_onelayer import Proposal
import math

def load_modules(pixel_dim, hidden_dim, latent_dim, LOAD_VERSION, CUDA, DEVICE):
    print('Loading EBM, proposal and optimizer...')
    ebm = Energy_function(out_channel=latent_dim, CUDA=CUDA, DEVICE=DEVICE)
    proposal = Proposal(latent_dim, hidden_dim, pixel_dim)
    if CUDA:
        ebm.cuda().to(DEVICE)   
        proposal.cuda().to(DEVICE)
#     ebm.load_state_dict(torch.load('../weights/ebm-%s' % LOAD_VERSION))
#     proposal.load_state_dict(torch.load('../weights/proposal-%s' % LOAD_VERSION))
    return ebm, proposal

def test_ebm_generation(ebm, proposal, sample_size):
    latents, _ = ebm.sample_priors(sample_size, 1, 1)
    images_ebm, _ = proposal(latents.squeeze(-2).squeeze(-2).squeeze(-2)) # S * latent_dim
    return images_ebm.cpu().detach()


def visual_samples_ebm(images):
    test_sample_size, patch_size2 = images.shape
    patch_size = int(math.sqrt(patch_size2))
    images = images.view(test_sample_size, patch_size, patch_size)
    gs = gridspec.GridSpec(1, test_sample_size)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 10 * test_sample_size))
    for i in range(test_sample_size):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
