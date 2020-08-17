import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ffebm.data import load_mnist
from ffebm.patches import load_mnist_patches
from ffebm.nets.mlp_encoder import Encoder
from ffebm.nets.mlp_decoder import Decoder

def load_modules(pixel_dim, hidden_dim, latent_dim, LOAD_VERSION, CUDA, DEVICE):
    print('Initialize encoder and decoder and optimizer...')
    enc = Encoder(latent_dim, hidden_dim, pixel_dim)
    dec = Decoder(latent_dim, hidden_dim, pixel_dim, CUDA, DEVICE)
    if CUDA:
        with torch.cuda.device(DEVICE):
            enc.cuda()  
            dec.cuda()
    print('Load trained weights...')
    enc.load_state_dict(torch.load('../weights/rws-mlp-enc-%s' % LOAD_VERSION))
    dec.load_state_dict(torch.load('../weights/rws-mlp-dec-%s' % LOAD_VERSION))
    return enc, dec

def test_mnist_one_batch(enc, dec, test_batch_size, test_sample_size, data_dir, CUDA, DEVICE):
    _, test_data = load_mnist(data_dir, test_batch_size)
    for (images, _) in test_data:
        break
    batch_size, _, pixel_size_sqrt, _ = images.shape
    images = images.squeeze(1).view(test_batch_size, pixel_size_sqrt*pixel_size_sqrt).repeat(test_sample_size, 1, 1)
    if CUDA:
        images = images.cuda().to(DEVICE)
    latents, _ = enc(images)
    _, recon, _ = dec(latents, images)
    recon = recon.cpu().detach().view(test_sample_size, test_batch_size, pixel_size_sqrt, pixel_size_sqrt).numpy()
    images = images.cpu().detach().view(test_sample_size, test_batch_size, pixel_size_sqrt, pixel_size_sqrt).numpy()
    return images, recon

def test_mnistpatch_one_batch(enc, dec, patch_size, test_batch_size, test_sample_size, data_dir, CUDA, DEVICE):
    data_dir = '../../../sebm_data/'
    train_path = data_dir+'MNIST_patch/train_data_%d_by_%d.pt' % (patch_size, patch_size)
    test_path = data_dir+'MNIST_patch/test_data_%d_by_%d.pt' % (patch_size, patch_size)
    train_data, test_data = load_mnist_patches((train_path, test_path), test_batch_size)    
    for images in train_data:
        break
    batch_size, _, pixel_size_sqrt, _ = images.shape
    images = images.squeeze(1).view(test_batch_size, pixel_size_sqrt*pixel_size_sqrt).repeat(test_sample_size, 1, 1)
    if CUDA:
        images = images.cuda().to(DEVICE)
    latents, _ = enc(images)
    _, recon, _ = dec(latents, images)
    recon = recon.cpu().detach().view(test_sample_size, test_batch_size, pixel_size_sqrt, pixel_size_sqrt).numpy()
    images = images.cpu().detach().view(test_sample_size, test_batch_size, pixel_size_sqrt, pixel_size_sqrt).numpy()
    return images, recon


def visual_samples_vae(images, recon):
    print("each row shows different samples of one MNIST image, where the true data image is at the rightmost columen ")
    test_sample_size, test_batch_size, _, _ = images.shape
    gs = gridspec.GridSpec(test_batch_size, test_sample_size+1)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
    fig = plt.figure(figsize=(10, 10 * test_batch_size / (1+test_sample_size)))
    for i in range(test_batch_size):
        for j in range(test_sample_size):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(recon[j, i], cmap='gray', vmin=0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[i, -1])
        ax.imshow(images[0, i], cmap='gray', vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])