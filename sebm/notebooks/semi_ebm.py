import torch
from torchvision import datasets, transforms
from sebm.data import load_data
from sebm.eval import *
from sebm.sgld import SGLD_sampler
from sebm.models import EBM

def iter_datasets(dataset):
    if dataset == 'mnist' or dataset =='fashionmnist':
        input_channels, im_height, im_width = 1, 28, 28
    else:
        input_channels, im_height, im_width = 3, 32, 32
    device = torch.device('cuda:%d' % torch.cuda.current_device())
    arch =  'simplenet'

    # lr, reg_alpha = 5e-5, 5e-3
    lr, reg_alpha = 1e-4, 1e-1
    # lr, reg_alpha = 5e-5, 5e-3
    optimize_priors = False
    if dataset == 'cifar10' or dataset == 'svhn':
        seed = 123
        channels, kernels, strides, paddings =[64,128,256,512], [3,4,4,4], [1,2,2,2], [1,1,1,1]
        hidden_dim, latent_dim, activation = [1024,128], 1, 'Swish'
    elif dataset == 'mnist' or dataset == 'fashionmnist':
        seed = 1
        channels, kernels, strides, paddings =[64,64,32,32], [3,4,4,4], [1,2,2,2], [1,1,1,1]
        hidden_dim, latent_dim, activation = [128,128], 1, 'Swish'
    else:
        raise NotImplementError
    data_noise_std = 3e-2
    sgld_noise_std, sgld_lr, sgld_num_steps = 7.5e-3, 2.0, 60
    buffer_init, buffer_dup_allowed = True, True
    data_dir = '../../../sebm_data/'
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
    ebm.load_state_dict(torch.load('../weights/final/cp-%s' % load_version)['model_state_dict'])

    evaluator = Evaluator_EBM(ebm, device, dataset, data_dir, data_noise_std=1e-2)    semi_nn_clf(model_name='vae', device=device, evaluator=evaluator, num_runs=10, num_epochs=100)
    semi_nn_clf(model_name='ebm', device=device, evaluator=evaluator, num_runs=10, num_epochs=100)
if __name__ == '__main__':
    l = ['mnist', 'fashionmnist', 'cifar10', 'svhn']
    for dataset in l:
        iter_datasets(dataset)
    