import math
import torch
import argparse
from cebm.data import setup_data_loader
from cebm.utils import set_seed, create_exp_name, init_models
from cebm.train.trainer import Trainer
from torch.distributions.normal import Normal

class Train_EBM_VERA(Trainer):
    def __init__(self, models, train_loader, device, exp_name, args):
        super().__init__(models, train_loader, args.num_epochs, device, exp_name)
#         self.image_noise_std = image_noise_std 
#         self.regularize_coeff = regularize_coeff
        optim_obj = getattr(torch.optim, args.optimizer)
        self.ebm_optimizer = optim_obj(self.models['ebm'].parameters(), lr=args.lr_p, betas=(args.beta1, args.beta2))
        self.gen_optimizer = optim_obj(self.models['gen'].parameters(), lr=args.lr_q, betas=(args.beta1, args.beta2))
        scheduler_kwargs = {
            "milestones": [int(epoch * len(train_loader)) for epoch in args.decay_epochs],
            "gamma": args.decay_rate
        }
        self.e_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.ebm_optimizer, **scheduler_kwargs)
        self.g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_optimizer, **scheduler_kwargs)        
        self.xee_optimizer = optim_obj(self.models['xee'].parameters(), lr=args.lr_xee)
        self.metric_names = ['E_div', 'E_data', 'E_model', 'ELBO']
        
        self.sample_size = args.sample_size
        self.gamma = args.gamma
        self.lambda_ent = args.lambda_ent
        self.min_x_logsgima = math.log(args.max_x_logsigma)
        self.max_x_logsigma = math.log(args.max_x_logsigma)

#         self.warmup_iters = args.warmup_iters
        
    def train_epoch(self, epoch):
        ebm = self.models['ebm']
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (x_data, _) in enumerate(self.train_loader):
#             images = (images + self.image_noise_std * torch.randn_like(images))
            x_data = x_data.to(self.device)
            z0, x_given_z0, _ = self.models['gen'].sample(x_data.shape[0])
            elbo, metric_epoch = self.elbo(z0, x_given_z0, metric_epoch)
            self.xee_optimizer.zero_grad()
            (-elbo).backward(retain_graph=True)
            self.xee_optimizer.step()
            loss, metric_epoch = self.cd(x_data, x_given_z0, metric_epoch)
            self.ebm_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.ebm_optimizer.step()
            loss = self.loss_gen(z0, x_given_z0)
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()
            self.models['gen'].x_logsigma.data.clamp_(min=self.min_x_logsgima, max=self.max_x_logsigma)
            if metric_epoch['E_div'].abs().item() > 1e+8:
                print('EBM diverging. Terminate training..')
                exit()
        return {k: (v / (b+1)).item() for k, v in metric_epoch.items()}

    def loss_gen(self, z0, x_given_z0):
        E_model = self.models['ebm'].energy(x_given_z0)
        z, log_xee = self.models['xee'].sample(z0=z0, sample_size=self.sample_size, detach_sigma=True)
#         xee_dist = Normal(z0, self.models['xee_logsigma'].detach().exp())
#         z = xee_dist.rsample((self.sample_size, ))
#         log_xee = xee_dist.log_prob(z).sum(-1)
        log_joint, x_mu_given_z = self.models['gen'].log_joint(x=x_given_z0, z=z)
        assert log_xee.shape == log_joint.shape
        w = torch.nn.functional.softmax(log_joint - log_xee, dim=0)
        grad_log_q_x_given_z = (x_given_z0[None] - x_mu_given_z) / (self.models['gen'].x_logsigma ** 2)
        grad_log_q_x = (w[:, :, None, None, None] * grad_log_q_x_given_z).sum(0).detach()
        assert grad_log_q_x.shape == x_given_z0.shape
        grad_eng_entropy = (grad_log_q_x * x_given_z0).flatten(start_dim=1).sum(1).mean(0)
        loss = E_model.mean() + self.lambda_ent * grad_eng_entropy
        return loss
    
    def elbo(self, z0, x_given_z0, metric_epoch):
        z, entropy = self.models['xee'].sample(z0=z0.detach(), entropy=True)
#         xee_dist = Normal(z0.detach(), self.models['xee_logsigma'].exp())
#         z = xee_dist.rsample()
        log_joint, _ = self.models['gen'].log_joint(x=x_given_z0, z=z)
#         entropy = xee_dist.entropy().sum(-1)
        elbo = (log_joint + entropy).mean()
        metric_epoch['ELBO'] += elbo.detach()
        return elbo, metric_epoch
        
    
    def cd(self, x_data, x_given_z0, metric_epoch):
        """
        maximize the log marginal i.e. log pi(x) = log \sum_{k=1}^K p(x, y=k)
        """
        x_data.requires_grad_()
        E_data = self.models['ebm'].energy(x_data)
        E_model = self.models['ebm'].energy(x_given_z0.detach())
        E_div = E_data.mean() - E_model.mean() 
        
        grad_E_data = torch.autograd.grad(E_data.sum(), x_data, retain_graph=True)[0].flatten(start_dim=1).norm(2, 1)
        loss = E_div + self.gamma * (grad_E_data ** 2).mean()
        metric_epoch['E_div'] += E_div.detach()
        metric_epoch['E_data'] += E_data.mean().detach()
        metric_epoch['E_model'] += E_model.mean().detach()
        return loss, metric_epoch

    
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    exp_name = create_exp_name(args)

    dataset_args = {'data': args.data, 
                    'data_dir': args.data_dir, 
                    'num_shots': -1,
                    'batch_size': args.batch_size,
                    'train': True, 
                    'normalize': False}
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
                    'dec_paddings': eval(args.dec_paddings),
                    'xee_init_sigma': args.xee_init_sigma,
                   }
    
    model_args = {'optimize_ib': args.optimize_ib,
                  'num_clusters': args.num_clusters}
    
    models = init_models(args.model_name, device, model_args, network_args)
    print('Start Training..')    
    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'device': device,
                    'exp_name': exp_name,
                    'args': args}
    trainer = Train_EBM_VERA(**trainer_args)
    trainer.train()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['CEBM_VERA', 'IGEBM_VERA', 'CEBM_GMM_VERA'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--exp_id', default=None)
    ## data config
    parser.add_argument('--data', required=True)
    parser.add_argument('--data_dir', default='../datasets/', type=str)
    parser.add_argument('--image_noise_std', default=1e-2, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--beta1', default=0., type=float)
    parser.add_argument('--beta2', default=0.9, type=float)
    parser.add_argument('--lr_p', default=1e-4, type=float)
    parser.add_argument('--lr_q', default=2e-4, type=float)
    parser.add_argument('--lr_xee', default=2e-4, type=float)
    parser.add_argument('--xee_init_sigma', default=0.1, type=float)
    parser.add_argument('--lambda_ent', default=1.0, type=float,
                        help="coefficient of grad entropy w.r.t. q_phi")
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help="coefficient of regularization of the grad E_data")
    parser.add_argument('--optimize_ib', default=False, action='store_true')
    parser.add_argument("--decay_epochs", nargs="+", type=float, default=[150, 175],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument('--warmup_iters', default=0, type=int)
    parser.add_argument('--max_x_logsigma', default=0.3, type=float)
    parser.add_argument('--min_x_logsigma', default=0.01, type=float)
    ## arch config 
    parser.add_argument('--channels', default="[32,32,64,64]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--hidden_dims', default="[256]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='SiLU')
    parser.add_argument('--dec_paddings', default="[1,1,0,0]")
    parser.add_argument('--leak_slope', default=0.1, type=float, help='parameter for LeakyReLU activation')
    parser.add_argument('--num_clusters', default=20, type=int)
    ## training config
    parser.add_argument('--sample_size', default=20, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    main(args)