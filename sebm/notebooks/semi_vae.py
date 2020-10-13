%matplotlib inline
import torch
from sebm.models import Encoder, Decoder
from torchvision import datasets, transforms
from sebm.data import load_data
def iter_datasets(dataset):

    if dataset == 'mnist' or dataset == 'fashionmnist':
        input_channels, im_height, im_width = 1, 28, 28
    else:
        input_channels, im_height, im_width = 3, 32, 32

    device = torch.device('cuda:0')
    arch =  'simplenet2' # 'mlp'
    lr = 1e-3
    seed = 1
    latent_dim = 128
    activation = 'ReLU'
    reparameterized = True
    heldout_class = -1
    load_version = 'vae-out=%s-d=%s-seed=%s-lr=%s-zd=%s-act=%s-arch=%s' % (heldout_class, dataset, seed, lr, latent_dim, activation, arch)
    data_dir = '/home/hao/Research/sebm_data/'
    if arch == 'simplenet2':
        if dataset == 'cifar10' or dataset == 'svhn':
            enc = Encoder(arch=arch,
                          reparameterized=reparameterized,
                          im_height=im_height, 
                          im_width=im_width, 
                          input_channels=input_channels, 
                          channels=[64,128,256,512], 
                          kernels=[3,4,4,4], 
                          strides=[1,2,2,2], 
                          paddings=[1,1,1,1], 
                          hidden_dim=[128],
                          latent_dim=latent_dim,
                          activation=activation)

            dec = Decoder(arch=arch,
                          device=device,
                          im_height=im_height, 
                          im_width=im_width, 
                          input_channels=input_channels, 
                          channels=[64,128,256,512], 
                          kernels=[3,4,4,4], 
                          strides=[1,2,2,2], 
                          paddings=[1,1,1,1], 
                          mlp_input_dim=latent_dim, ## TODO: hand-coded for now
                          hidden_dim=[128],
                          mlp_output_dim=8192,
                          activation=activation)
        elif dataset == 'mnist' or dataset == 'fashionmnist':
            enc = Encoder(arch=arch,
                          reparameterized=reparameterized,
                          im_height=im_height, 
                          im_width=im_width, 
                          input_channels=input_channels, 
                          channels=[64,64,32,32], 
                          kernels=[3,4,4,4], 
                          strides=[1,2,2,2], 
                          paddings=[1,1,1,1], 
                          hidden_dim=[128],
                          latent_dim=latent_dim,
                          activation=activation)
            dec = Decoder(arch=arch,
                          device=device,
                          im_height=im_height, 
                          im_width=im_width, 
                          input_channels=input_channels, 
                          channels=[64,64,32,32], 
                          kernels=[3,4,4,4], 
                          strides=[1,2,2,2], 
                          paddings=[0,0,1,1], 
                          mlp_input_dim=latent_dim, ## TODO: hand-coded for now
                          hidden_dim=[128],
                          mlp_output_dim=288,
                          activation=activation)
        else:
            raise NotImplementError

    else:
        raise NotImplementError

    enc = enc.cuda().to(device)  
    dec = dec.cuda().to(device)
    print('Loading trained models...')
    enc.load_state_dict(torch.load('../weights/final/cp-%s' % load_version)['enc_state_dict'])
    dec.load_state_dict(torch.load('../weights/final/cp-%s' % load_version)['dec_state_dict'])

    from sebm.eval import *
    evaluator = Evaluator_VAE(enc, dec, arch, device, dataset, data_dir)
    semi_nn_clf(model_name='vae', device=device, evaluator=evaluator, num_runs=10, num_epochs=100)
    
if __name__ == '__main__':
    l = ['mnist', 'fashionmnist', 'cifar10', 'svhn']
    for dataset in l:
        iter_datasets(dataset)