import torch
from sebm.models import Encoder, Decoder
from torchvision import datasets, transforms
from sebm.data import load_data
from sebm.eval import *
from sebm.sgld import SGLD_sampler
from sebm.models import EBM
from sebm.cebm_sgld import init_cebm
from sebm.cebm_gmm_sgld import CEBM_GMM_2ss

def init_model(model, dataset):
    if dataset == 'mnist' or dataset == 'fashionmnist':
        input_channels, im_height, im_width = 1, 28, 28
    else:
        input_channels, im_height, im_width = 3, 32, 32
    device = torch.device('cuda:1')

    if model == 'vae':
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
        enc.load_state_dict(torch.load('weights/cp-%s' % load_version)['enc_state_dict'])
        dec.load_state_dict(torch.load('weights/cp-%s' % load_version)['dec_state_dict'])
        evaluator = Evaluator_VAE(enc, dec, arch, device, dataset, data_dir)
        
    elif model == 'igebm':
        arch =  'simplenet'
        seed = 1
        optimize_priors = False
        data_noise_std = 3e-2
        sgld_noise_std, sgld_lr, sgld_num_steps = 7.5e-3, 2.0, 60
        buffer_init, buffer_dup_allowed = True, True
        data_dir = '/home/hao/Research/sebm_data/'
        if dataset == 'mnist':
            lr, reg_alpha = 5e-5, 5e-3
            channels, kernels, strides, paddings =[64,64,32,32], [3,4,4,4], [1,2,2,2], [1,1,1,1]
            hidden_dim, latent_dim, activation = [128,128], 1, 'Swish'
        elif dataset == 'cifar10':
            lr, reg_alpha = 5e-5, 5e-2
            channels, kernels, strides, paddings =[64,128,256,512], [3,4,4,4], [1,2,2,2], [1,1,1,1]
            hidden_dim, latent_dim, activation = [1024,128], 1, 'Swish'
        else:
            raise NotImplementError
        load_version = 'igebm-d=%s-seed=%d-lr=%s-zd=%d-d_ns=%s-sgld-ns=%s-lr=%s-steps=%s-reg=%s-act=%s-arch=%s' % (dataset, seed, lr, latent_dim, data_noise_std, sgld_noise_std, sgld_lr, sgld_num_steps,  reg_alpha, activation, arch)
        ebm = EBM(arch=arch,
                        im_height=im_height, 
                        im_width=im_width, 
                        input_channels=input_channels, 
                        channels=channels, 
                        kernels=kernels, 
                        strides=strides, 
                        paddings=paddings, 
                        hidden_dim=hidden_dim,
                        latent_dim=latent_dim,
                        activation=activation)
        ebm = ebm.cuda().to(device)
        print('Loading trained weights..')
        ebm.load_state_dict(torch.load('weights/final/cp-%s' % load_version)['model_state_dict'])
        evaluator = Evaluator_EBM(ebm, device, dataset, data_dir, data_noise_std=1e-2)

    elif model == 'cebm':
        arch =  'simplenet2' # 'wresnet' # 'simplenet'
        ss = 2
        seed = 1
        optimize_priors = False
        data_noise_std = 3e-2
        sgld_noise_std, sgld_lr, sgld_num_steps = 7.5e-3, 2.0, 60
        buffer_init, buffer_dup_allowed = True, True
        heldout = -1
        data_dir = '/home/hao/Research/sebm_data/'
        if dataset == 'mnist':
            lr, reg_alpha = 1e-4, 1e-1
            channels, kernels, strides, paddings =[64,64,32,32], [3,4,4,4], [1,2,2,2], [1,1,1,1]
            hidden_dim, latent_dim, activation = [128], 128, 'Swish'
        elif dataset == 'cifar10':
            lr, reg_alpha = 1e-4, 1e-1
            channels, kernels, strides, paddings =[64,128,256,512], [3,4,4,4], [1,2,2,2], [1,1,1,1]
            hidden_dim, latent_dim, activation = [1024], 128, 'Swish'
        else:
            raise NotImplementError
        load_version = 'cebm_%sss-out=%s-d=%s-seed=%d-lr=%s-zd=%d-d_ns=%s-sgld-ns=%s-lr=%s-steps=%s-reg=%s-act=%s-arch=%s' % (ss, heldout, dataset, seed, lr, latent_dim, data_noise_std, sgld_noise_std, sgld_lr, sgld_num_steps,  reg_alpha, activation, arch)
        if arch == 'simplenet2':
            ebm = init_cebm(arch=arch,
                            ss=ss,
                            optimize_priors=optimize_priors,
                            device=device,
                            im_height=im_height, 
                            im_width=im_width, 
                            input_channels=input_channels, 
                            channels=channels, 
                            kernels=kernels, 
                            strides=strides, 
                            paddings=paddings, 
                            hidden_dim=hidden_dim,
                            latent_dim=latent_dim,
                            activation=activation)
        ebm = ebm.cuda().to(device)
        print('Loading trained weights..')
        ebm.load_state_dict(torch.load('weights/final/cp-%s' % load_version)['model_state_dict'])
        evaluator = Evaluator_EBM(ebm, device, dataset, data_dir, data_noise_std=1e-2)
        
    elif model == 'cebm_gmm':
        arch =  'simplenet2' # 'wresnet' # 'simplenet'
        ss = 2
        seed = 1
        optimize_priors = False
        data_noise_std = 3e-2
        sgld_noise_std, sgld_lr, sgld_num_steps = 7.5e-3, 2.0, 60
        buffer_init, buffer_dup_allowed = True, True
        data_dir = '/home/hao/Research/sebm_data/'
        if dataset == 'mnist':
            lr, reg_alpha = 1e-4, 1e-1
            channels, kernels, strides, paddings =[64,64,32,32], [3,4,4,4], [1,2,2,2], [1,1,1,1]
            hidden_dim, latent_dim, activation = [128], 128, 'Swish'
        elif dataset == 'fashionmnist':
            lr, reg_alpha = 1e-4, 1e-1
            channels, kernels, strides, paddings =[64,64,32,32], [3,4,4,4], [1,2,2,2], [1,1,1,1]
            hidden_dim, latent_dim, activation = [128], 128, 'Swish'
        elif dataset == 'cifar10':
            lr, reg_alpha = 1e-4, 1e-1
            channels, kernels, strides, paddings =[64,128,256,512], [3,4,4,4], [1,2,2,2], [1,1,1,1]
            hidden_dim, latent_dim, activation = [1024], 128, 'Swish'
        else:
            raise NotImplementError
        load_version = 'cebm_gmm_2ss-out=-1-d=%s-seed=%d-lr=%s-zd=%d-d_ns=%s-sgld-ns=%s-lr=%s-steps=%s-reg=%s-act=%s-arch=%s' % (dataset, seed, lr, latent_dim, data_noise_std, sgld_noise_std, sgld_lr, sgld_num_steps,  reg_alpha, activation, arch)
        if arch == 'simplenet2':
            ebm = CEBM_GMM_2ss(K=10,
                            arch=arch,
                            optimize_priors=optimize_priors,
                            device=device,
                            im_height=im_height, 
                            im_width=im_width, 
                            input_channels=input_channels, 
                            channels=channels, 
                            kernels=kernels, 
                            strides=strides, 
                            paddings=paddings, 
                            hidden_dim=hidden_dim,
                            latent_dim=latent_dim,
                            activation=activation)
        ebm = ebm.cuda().to(device)
        print('Loading trained weights..')
        weights = torch.load('weights/cp-%s' % load_version)
        ebm.load_state_dict(weights['model_state_dict'])
        ebm.prior_nat1 = weights['prior_nat1'].to(device)
        ebm.prior_nat2 = weights['prior_nat2'].to(device)
        evaluator = Evaluator_EBM_GMM(ebm, device, dataset, data_dir, data_noise_std=1e-2)
    return evaluator

if __name__ == "__main__":
    import torch
    import argparse
    from tqdm import tqdm
    import math
    parser = argparse.ArgumentParser('Label Propgations')
    parser.add_argument('--model', choices=['vae', 'igebm', 'cebm', 'cebm_gmm'])
    parser.add_argument('--dataset', choices=['mnist', 'fashionmnist', 'cifar10'])
#     parser.add_argument('--algo_name', choice=['lp', 'ls'])
#     parser.add_argument('--seed', default=1, type=int)
#     parser.add_argument('--gamma', default=20, type-float)
#     parser.add_argument('--max_iter', default=30, type=int)
#     parser.add_argument('--kernel')
    args = parser.parse_args()
    evaluator = init_model(model=args.model, dataset=args.dataset)
    NUM_SHOTs = [1, 10, 100]
    NUM_NEIGHBORs = [5]
    kernel = 'knn'
    algo_name, seed, gamma, max_iter = 'ls', 123, 20, 30    
    for num_shots in tqdm(NUM_SHOTs):
        for n_neighbors in NUM_NEIGHBORs:
            accu = label_propagation(algo_name, evaluator, num_shots, seed, kernel, gamma, n_neighbors, max_iter)
            fout = open('label_propagation_accuracy.txt', 'a+')
            print('model=%s, data=%s, accuracy=%s, algo=%s, seed=%d, kernel=%s, gamma=%s, max_iter=%d, num_shots=%d, n_neighbors=%d' % (args.model, args.dataset, accu, algo_name, seed, kernel, gamma, max_iter, num_shots, n_neighbors), file=fout)
            fout.close()