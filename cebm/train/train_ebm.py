import torch
import argparse
from cebm.data import setup_data_loader
from cebm.sgld import SGLD_Sampler
from cebm.utils import set_seed, create_exp_name, init_models
from cebm.train.trainer import Trainer
from torch.distributions.normal import Normal

class Train_EBM(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, sgld_sampler, optimizer, lr, sgld_steps, image_noise_std, regularize_coeff):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.sgld_sampler = sgld_sampler 
        self.sgld_steps = sgld_steps
        self.image_noise_std = image_noise_std 
        self.regularize_coeff = regularize_coeff
        if optimizer == 'Adam':
            self.optimizer = getattr(torch.optim, optimizer)(list(self.models['ebm'].parameters()), lr=lr)
        else:
            self.optimizer = getattr(torch.optim, optimizer)(list(self.models['ebm'].parameters()), lr=lr)
        self.metric_names = ['E_div', 'E_data', 'E_model']

    def train_epoch(self, epoch):
        ebm = self.models['ebm']
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (images, _) in enumerate(self.train_loader):
            self.optimizer.zero_grad() 
            images = (images + self.image_noise_std * torch.randn_like(images)).to(self.device)
            loss, metric_epoch = self.loss(ebm, images, metric_epoch)
                
            if loss.abs().item() > 1e+8:
                print('EBM diverging. Terminate training..')
                exit()
            loss.backward()
            self.optimizer.step()
        return {k: (v / (b+1)).item() for k, v in metric_epoch.items()}

    
    def loss(self, ebm, data_images, metric_epoch, pcd=True, init_samples=None):
        """
        maximize the log marginal i.e. log pi(x) = log \sum_{k=1}^K p(x, y=k)
        """
        E_data = ebm.energy(data_images)
        simulated_images = self.sgld_sampler.sample(ebm, len(data_images), self.sgld_steps, pcd=pcd, init_samples=init_samples)
        E_model = ebm.energy(simulated_images)
        z_mu, z_sigma = ebm.latent_params(data_images)
        E_div = E_data.mean() - E_model.mean() 
        loss = E_div + self.regularize_coeff * (E_data**2).mean()
        metric_epoch['E_div'] += E_div.detach()
        metric_epoch['E_data'] += E_data.mean().detach()
        metric_epoch['E_model'] += E_model.mean().detach()
        return loss, metric_epoch
    
#     def loss2(self, ebm, data_images, metric_epoch, pcd=True, sample_size=1):
#         post_mu, post_std = ebm.latent_params(data_images)
#         p = Normal(post_mu, post_std)
#         if sample_size == 1:
#             data_z = p.sample() # B * D
#         else:
#             data_z = p.sample((sample_size,)) # S * B * D
#         data_lf = ebm.log_factor(data_images, data_z, expand_dim=None if sample_size==1 else sample_size)
#         negative_images = self.sgld_sampler.cond_sample(ebm, data_z, len(data_images), self.sgld_steps, pcd=pcd)
#         negative_lf = ebm.log_factor(negative_images, data_z, expand_dim=None if sample_size==1 else sample_size)
#         loss = negative_lf.mean() - data_lf.mean() + self.regularize_coeff * (negative_lf**2 + data_lf**2).mean()
#         E_data = ebm.energy(data_images).mean().detach()
#         E_model = ebm.energy(negative_images).mean().detach()
#         metric_epoch['E_data'] += E_data
#         metric_epoch['E_model'] += E_model
#         metric_epoch['E_div'] += E_data - E_model
#         metric_epoch['LF_data'] += data_lf.mean().detach()
#         metric_epoch['LF_model'] += negative_lf.mean().detach()
# #         breakpoint()
#         return loss, metric_epoch
    
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    exp_name = create_exp_name(args)

    dataset_args = {'data': args.data, 
                    'data_dir': args.data_dir, 
                    'num_shots': -1,
                    'batch_size': args.batch_size,
                    'train': True, 
                    'normalize': True}
    train_loader, im_h, im_w, im_channels = setup_data_loader(**dataset_args)
    
    network_args = {'device': device,
                    'im_height': im_h, 
                    'im_width': im_w, 
                    'input_channels': im_channels, 
                    'channels': eval(args.channels), 
                    'kernels': eval(args.kernels), 
                    'strides': eval(args.strides), 
                    'paddings': eval(args.paddings),
                    'hidden_dims': eval(args.hidden_dims),
                    'latent_dim': args.latent_dim,
                    'activation': args.activation,
                    'arch': args.arch,
                   }
    
    model_args = {'optimize_ib': args.optimize_ib,
                  'num_clusters': args.num_clusters}
    
    models = init_models(args.model_name, device, model_args, network_args)
    
    print('Start Training..')
    sgld_args = {'im_h': im_h, 
                 'im_w': im_w, 
                 'im_channels': im_channels,
                 'device': device,
                 'alpha': args.sgld_alpha,
                 'noise_std': args.sgld_noise_std,
                 'buffer_size': args.buffer_size,
                 'reuse_freq': args.reuse_freq}
    
    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'sgld_sampler': SGLD_Sampler(**sgld_args),
                    'optimizer': args.optimizer,
                    'lr': args.lr,
                    'sgld_steps': args.sgld_steps,
                    'image_noise_std': args.image_noise_std,
                    'regularize_coeff': args.regularize_coeff}
    trainer = Train_EBM(**trainer_args)
    trainer.train()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['CEBM', 'CEBM_GMM', 'IGEBM'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--exp_id', default=None)
    ## data config
    parser.add_argument('--data', required=True)
    parser.add_argument('--data_dir', default='../datasets/', type=str)
    parser.add_argument('--image_noise_std', default=3e-2, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['AdamW', 'Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--optimize_ib', default=False, action='store_true')
    ## arch config 
    parser.add_argument('--channels', default="[32,32,64,64]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--hidden_dims', default="[256]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='Swish')
    parser.add_argument('--leak_slope', default=0.1, type=float, help='parameter for LeakyReLU activation')
    parser.add_argument('--num_clusters', default=20, type=int)
    parser.add_argument('--arch', default='4cnn', choices=['simplenet', '4cnn'])
    ## training config
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=100, type=int)

    ## sgld sampler config
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--reuse_freq', default=0.95, type=float)
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_alpha', default=2.0, type=float, help='step size is half of this value')
    parser.add_argument('--sgld_steps', default=40, type=int)
    parser.add_argument('--regularize_coeff', default=1e-1, type=float)   
    
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    main(args)