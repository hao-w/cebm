import torch
import torch.nn as nn
import argparse
from cebm.data import setup_omniglot_loader
from cebm.utils import set_seed, create_exp_name, init_models, save_models
from cebm.train.trainer import Trainer
from cebm.sgld import SGLD_Sampler_GMM
from tqdm import tqdm

class Train_MetaCEBM(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, sgld_sampler, optimizer, lr, gibbs_sweeps, sample_size, sgld_steps, image_noise_std, regularize_coeff):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.sgld_sampler = sgld_sampler 
        self.optimizer = getattr(torch.optim, optimizer)(list(self.models['ebm'].parameters()), lr=lr)
        self.gibbs_sweeps = gibbs_sweeps
        self.sample_size = sample_size
        self.sgld_steps = sgld_steps
        self.image_noise_std = image_noise_std 
        self.regularize_coeff = regularize_coeff
        self.metric_names = ['E_div', 'E_data', 'E_model']

    def train_epoch(self, epoch):
        ebm = self.models['ebm']
        train_episodes = self.train_loader.generate_tasks()
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, images in enumerate(train_episodes):
            self.optimizer.zero_grad()
#             images = images.repeat(self.sample_size, 1, 1, 1, 1).to(self.device)
            images = (images + self.image_noise_std * torch.randn_like(images)).to(self.device)
            loss, metric_epoch = self.loss(ebm, images, metric_epoch)
            loss.backward()
            self.optimizer.step()
            
        return {k: (v / (b+1)).item() for k, v in metric_epoch.items()}
    
    def loss(self, ebm, data_images, metric_epoch):
        """
        maximize the log marginal i.e. log pi(x) = log \sum_{k=1}^K p(x, y=k)
        """
        alpha, beta, mu, nu, pi, gammas, c_means, c_stds, ys = ebm.gibbs_updates(data_images, 
                                                                                 self.gibbs_sweeps, 
                                                                                 self.sample_size)
        E_data = ebm.energy(data_images, c_means, c_stds, ys)
        
        simulated_images = self.sgld_sampler.sample(ebm, 
                                                    len(data_images), 
                                                    self.sgld_steps, 
                                                    c_means, 
                                                    c_stds, 
                                                    ys)
        
        E_model = ebm.energy(simulated_images, c_means, c_stds, ys)
        E_div = E_data.mean() - E_model.mean() 
        loss = E_div + self.regularize_coeff * (E_data**2).mean()
        metric_epoch['E_div'] += E_div.detach()
        metric_epoch['E_data'] += E_data.mean().detach()
        metric_epoch['E_model'] += E_model.mean().detach()
        return loss, metric_epoch

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.data == 'omniglot':
        dataset_args = {'data_dir': args.data_dir,
                        'way': args.way,
                        'shot': args.shot,
                        'train': True,
                        'normalize': True}
        task_loader, im_h, im_w, im_channels = setup_omniglot_loader(**dataset_args)
        
    network_args = {'num_clusters': args.way,
                    'device': device,
                    'im_height': im_h,
                    'im_width': im_w,
                    'input_channels': im_channels,
                    'channels': eval(args.channels),
                    'kernels': eval(args.kernels),
                    'strides': eval(args.strides),
                    'paddings': eval(args.paddings),
                    'latent_dim': args.latent_dim,
                    'activation': args.activation}
    
    model_args = []
    models = init_models(args.model_name, device, model_args, network_args)
    for k,v in models.items():
        print(v)
    exp_name = create_exp_name(args)
    
    sgld_args = {'im_h': im_h, 
                 'im_w': im_w, 
                 'im_channels': im_channels,
                 'device': device,
                 'alpha': args.sgld_alpha,
                 'noise_std': args.sgld_noise_std,
                 'buffer_size': args.buffer_size,
                 'reuse_freq': args.reuse_freq}
    sgld_sampler  = SGLD_Sampler_GMM(**sgld_args)
    
    print('Start Training..')
    trainer_args = {'models': models,
                    'train_loader': task_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'sgld_sampler': sgld_sampler,
                    'optimizer': args.optimizer,
                    'lr': args.lr,
                    'gibbs_sweeps': args.gibbs_sweeps,
                    'sample_size': args.sample_size,
                    'sgld_steps': args.sgld_steps,
                    'image_noise_std': args.image_noise_std,
                    'regularize_coeff': args.regularize_coeff}
    trainer = Train_MetaCEBM(**trainer_args)

    trainer.train()
        
def parse_args():
    parser = argparse.ArgumentParser('META_CEBM')
    parser.add_argument('--model_name', default='META_CEBM')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--exp_id', default=None)
    # data config
    parser.add_argument('--data', default='omniglot', choices=['omniglot'])
    parser.add_argument('--data_dir', default='../datasets/', type=str)
    # training config
    parser.add_argument('--image_noise_std', default=0.0, type=float)
    parser.add_argument('--sample_size', default=1, type=int)
    parser.add_argument('--way', default=5, type=int)
    parser.add_argument('--shot', default=20, type=int)
    parser.add_argument('--gibbs_sweeps', default=10, type=int)
    # arch config
    parser.add_argument('--channels', default="[64,64,64,64]")
    parser.add_argument('--kernels', default="[3,3,3,3]")
    parser.add_argument('--strides', default="[1,1,1,1]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--activation', default='ReLU')    
    # optim config
    parser.add_argument('--optimizer', choices=['Adam'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    # training config
    parser.add_argument('--num_epochs', default=200, type=int)
#     parser.add_argument('--batch_size', default=100, type=int)
    # sgld config
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--reuse_freq', default=0.95, type=float)
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_alpha', default=2.0, type=float, help='step size is half of this value')
    parser.add_argument('--sgld_steps', default=60, type=int)
    parser.add_argument('--regularize_coeff', default=1e-1, type=float)   
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
