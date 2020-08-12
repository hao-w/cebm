import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ffebm.data import load_mnist
# from ffebm.data_noise import DATA_NOISE_sampler
from ffebm.nets.ffebm_multilayers import Energy_function
from ffebm.nets.proposal_onelayer import Proposal
import math

def load_modules(patch_sizes, hidden_dim, latent_dims, LOAD_VERSION, CUDA, DEVICE):
    print('Initialize EBM, proposal and optimizer...')
    (latent_dim1, latent_dim2, latent_dim3) = latent_dims
    (patch_size1, patch_size2, patch_size3) = patch_sizes
    ebm1 = Energy_function(in_channel=1, out_channel=latent_dim1, kernel_size=4, stride=4, padding=0, CUDA=CUDA, DEVICE=DEVICE)
    ebm2 = Energy_function(in_channel=latent_dim1, out_channel=latent_dim2, kernel_size=3, stride=3, padding=1, CUDA=CUDA, DEVICE=DEVICE)
    ebm3 = Energy_function(in_channel=latent_dim3, out_channel=latent_dim3, kernel_size=3, stride=1, padding=0, CUDA=CUDA, DEVICE=DEVICE)
    proposal1 = Proposal(latent_dim1, hidden_dim, patch_size1**2, in_channel=1)
    proposal2 = Proposal(latent_dim2, hidden_dim, patch_size2**2, in_channel=32)
    proposal3 = Proposal(latent_dim3, hidden_dim, patch_size3**2, in_channel=32)
    if CUDA:
        with torch.cuda.device(DEVICE):
            ebm1.cuda()
            ebm2.cuda()
            ebm3.cuda()
            proposal1.cuda()
            proposal2.cuda()
            proposal3.cuda()
    ebm1.load_state_dict(torch.load('../weights/ebm1-%s' % LOAD_VERSION))
    ebm2.load_state_dict(torch.load('../weights/ebm2-%s' % LOAD_VERSION))
    ebm3.load_state_dict(torch.load('../weights/ebm3-%s' % LOAD_VERSION))
    proposal1.load_state_dict(torch.load('../weights/proposal1-%s' % LOAD_VERSION))
    proposal2.load_state_dict(torch.load('../weights/proposal2-%s' % LOAD_VERSION))
    proposal3.load_state_dict(torch.load('../weights/proposal3-%s' % LOAD_VERSION))
    return (ebm1, ebm2, ebm3), (proposal1, proposal2, proposal3) 

def test_ebm_generation(ebms, proposals, sample_size, CUDA, DEVICE):
    (ebm1, ebm2, ebm3) = ebms 
    (proposal1, proposal2, proposal3) = proposals
    latent3, _ = ebm3.sample_priors(sample_size=sample_size, batch_size=1, num_patches=1)
    images_ebm3, _ = proposal3(latent3)
    S, B, P, _, in_channel, patch_dim2 = images_ebm3.shape
    patch_dim = int(math.sqrt(patch_dim2))
    images_ebm3 = images_ebm3.view(S, B, P, P, in_channel, patch_dim, patch_dim).squeeze(2).squeeze(2).permute(0, 1, 3, 4, 2)
    images_ebm2, _ = proposal2(images_ebm3)
    S, B, P, _, in_channel, patch_dim2 = images_ebm2.shape
    patch_dim = int(math.sqrt(patch_dim2))
    images_ebm2 = images_ebm2.view(S, B, P, P, in_channel, patch_dim, patch_dim).permute(0, 1, 2, 3, 5, 6, 4)
    latent1 = torch.zeros((S, B, patch_dim*P, patch_dim*P, in_channel))
    if CUDA:
        latent1 = latent1.cuda().to(DEVICE)
    for i in range(P):
        for j in range(P):
            latent1[:, :, i:i+patch_dim, j:j+patch_dim, :] = images_ebm2[:, :, i, j, :, :, :]
    latent1 = latent1[:,:,1:-1, 1:-1, :]
    images_ebm1, _ = proposal1(latent1)
    S, B, P, _, in_channel, patch_dim2 = images_ebm1.shape
    patch_dim = int(math.sqrt(patch_dim2))
    images_ebm1 = images_ebm1.view(S, B, P, P, in_channel, patch_dim, patch_dim).permute(0, 1, 2, 3, 5, 6, 4)
    images_final = torch.zeros((S, B, patch_dim*P, patch_dim*P, in_channel))
    if CUDA:
        images_final = images_final.cuda().to(DEVICE)
    for i in range(P):
        for j in range(P):
            images_final[:, :, i:i+patch_dim, j:j+patch_dim, :] = images_ebm1[:, :, i, j, :, :, :]
    
    return images_final.squeeze(-1).cpu().detach()

def visual_samples_ebm(images):
    test_sample_size, patch_size, _ = images.shape
    gs = gridspec.GridSpec(1, test_sample_size)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 10 * test_sample_size))
    for i in range(test_sample_size):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
