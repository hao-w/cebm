import math
import torch
import argparse
from cebm.data import setup_data_loader
from cebm.utils import set_seed, create_exp_name, init_models
from cebm.train.trainer import Trainer
from torch.distributions.normal import Normal
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms

# writer = SummaryWriter()

class Train_EBM_VERA(Trainer):
    def __init__(self, models, train_loader, device, exp_name, args):
        super().__init__(models, train_loader, args.num_epochs, device, exp_name)
#         self.image_noise_std = image_noise_std 
#         self.regularize_coeff = regularize_coeff
        optim_obj = getattr(torch.optim, args.optimizer)
        self.ebm_optimizer = optim_obj(self.models['ebm'].parameters(), lr=args.lr_p, betas=(args.beta1, args.beta2))
        self.gen_optimizer = optim_obj(self.models['gen'].parameters(), lr=args.lr_q, betas=(args.beta1, args.beta2))
        self.xee_optimizer = optim_obj(self.models['xee'].parameters(), lr=args.lr_xee)

        scheduler_kwargs = {
            "milestones": [int(epoch * len(train_loader)) for epoch in args.decay_epochs],
            "gamma": args.decay_rate
        }
        self.lr_p = args.lr_p
        self.lr_q = args.lr_q
        self.e_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.ebm_optimizer, **scheduler_kwargs)
        self.g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_optimizer, **scheduler_kwargs)        

        self.metric_names = ['loss_p', 'loss_q', 'ELBO', 'E_data', 'E_model']
        
        self.sample_size = args.sample_size
        self.gamma = args.gamma
        self.lambda_ent = args.lambda_ent
        self.min_x_logsgima = math.log(args.min_x_logsigma)
        self.max_x_logsigma = math.log(args.max_x_logsigma)
        self.warmup_iters = args.warmup_iters
        self.likelihood = args.likelihood
        
    def train_epoch(self, epoch):
        ebm = self.models['ebm']
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (x_data, _) in enumerate(self.train_loader):
            # warmup lr
            itr = epoch * len(self.train_loader) + b + 1
            if itr < self.warmup_iters:
                lr_p = self.lr_p * (itr + 1) / float(self.warmup_iters)
                lr_q = self.lr_q * (itr + 1) / float(self.warmup_iters)
                for param_group in self.ebm_optimizer.param_groups:
                    param_group['lr'] = lr_p
                for param_group in self.gen_optimizer.param_groups:
                    param_group['lr'] = lr_q
                    
            x_data = x_data.to(self.device)
            z0, x_given_z0, _ = self.models['gen'].sample(x_data.shape[0])
            
            loss_ebm, metric_epoch = self.cd(x_data, x_given_z0, metric_epoch)
            self.ebm_optimizer.zero_grad()
            loss_ebm.backward()
#             breakpoint()
            self.ebm_optimizer.step()
            
            elbo, metric_epoch = self.elbo(z0, x_given_z0, metric_epoch)
            self.xee_optimizer.zero_grad()
            (-elbo).backward()
#             breakpoint()
            self.xee_optimizer.step()
            
            
            loss_gen, metric_epoch = self.loss_gen(z0, x_given_z0, metric_epoch)
            self.gen_optimizer.zero_grad()
            loss_gen.backward()
#             breakpoint()
            self.gen_optimizer.step()
    
            if self.likelihood == 'gaussian':
                self.models['gen'].x_logsigma.data.clamp_(min=self.min_x_logsgima, max=self.max_x_logsigma)

            self.e_lr_scheduler.step()
            self.g_lr_scheduler.step()
            
            if loss_ebm.detach().abs().item() > 1e+8:
                print('EBM diverging. Terminate training..')
                exit()
        return {k: (v / (b+1)).item() for k, v in metric_epoch.items()}

    def loss_gen(self, z0, x_given_z0, metric_epoch):
        E_model = self.models['ebm'].energy(x_given_z0)
        z, log_xee = self.models['xee'].sample(z0=z0, sample_size=self.sample_size, detach_sigma=True)
        log_joint, x_mu_given_z = self.models['gen'].log_joint(x=x_given_z0, z=z)
            
        if self.likelihood == 'gaussian':
            neg_grad_log_q_x_given_z = (x_given_z0[None] - x_mu_given_z) / (self.models['gen'].x_logsigma.exp() ** 2)
            
        elif self.likelihood == 'cb':
            x_expand = x_given_z0.detach().expand(self.sample_size, *x_given_z0.shape).requires_grad_()
            neg_grad_log_q_x_given_z = torch.autograd.grad(self.models['gen'].ll(x=x_expand, z=z.detach()).sum(), 
                                                            x_expand)[0]
            
        assert log_xee.shape == log_joint.shape
        w = torch.nn.functional.softmax(log_joint - log_xee, dim=0).detach()   
        neg_grad_log_q_x = (w[:, :, None, None, None] * neg_grad_log_q_x_given_z).sum(0).detach()
        assert neg_grad_log_q_x.shape == x_given_z0.shape
        grad_entropy = (neg_grad_log_q_x * x_given_z0).flatten(start_dim=1).sum(1).mean(0)
        loss = E_model.mean() - self.lambda_ent * grad_entropy
        metric_epoch['loss_q'] += loss.detach() #(1. / (w**2).sum(0)).mean()
        return loss, metric_epoch
    
    def elbo(self, z0, x_given_z0, metric_epoch):
        z, entropy = self.models['xee'].sample(z0=z0.detach(), entropy=True)
#         xee_dist = Normal(z0.detach(), self.models['xee_logsigma'].exp())
#         z = xee_dist.rsample()
        log_joint, _ = self.models['gen'].log_joint(x=x_given_z0.detach(), z=z)
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
        
        grad_E_data = torch.autograd.grad(E_data.sum(), x_data, create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
        loss = E_div + \
                self.gamma * (grad_E_data ** 2. / 2.).mean()
#                 self.gamma * ((E_data**2).mean() + (E_model**2).mean())
        metric_epoch['loss_p'] += loss.detach()
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
                    'normalize': True if args.likelihood=="gaussian" else False}
    
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
                    'leak_slope': args.leak_slope,
                    'gen_channels': eval(args.gen_channels), 
                    'gen_kernels': eval(args.gen_kernels), 
                    'gen_strides': eval(args.gen_strides), 
                    'gen_paddings': eval(args.gen_paddings),
                    'gen_activation': args.gen_activation,
                    'output_arch': args.output_arch,
                    'likelihood': args.likelihood,
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
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--exp_id', default=None)
    ## data config
    parser.add_argument('--data', required=True)
    parser.add_argument('--data_dir', default='../datasets/', type=str)
    parser.add_argument('--image_noise_std', default=1e-2, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--beta1', default=0., type=float)
    parser.add_argument('--beta2', default=0.9, type=float)
    parser.add_argument('--lr_p', default=5e-5, type=float)
    parser.add_argument('--lr_q', default=2e-4, type=float)
    parser.add_argument('--lr_xee', default=2e-4, type=float)
    parser.add_argument('--xee_init_sigma', default=0.1, type=float)
    parser.add_argument('--lambda_ent', default=1.0, type=float,
                        help="coefficient of grad entropy w.r.t. q_phi")
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help="coefficient of regularization of the grad E_data")
    parser.add_argument('--optimize_ib', default=False, action='store_true')
    parser.add_argument("--decay_epochs", nargs="+", type=float, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument('--warmup_iters', default=0, type=int)
    parser.add_argument('--max_x_logsigma', default=0.3, type=float)
    parser.add_argument('--min_x_logsigma', default=0.01, type=float)
    ## arch config 
    parser.add_argument('--output_arch', default='mlp')
    parser.add_argument('--channels', default="[32,32,64,64]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--hidden_dims', default="[256]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='SiLU')
    parser.add_argument('--dec_paddings', default="[1,1,0,0]")
    parser.add_argument('--leak_slope', default=0.2, type=float, help='parameter for LeakyReLU activation')
    parser.add_argument('--gen_kernels', default="[4,4,3,4,4]")
    parser.add_argument('--gen_channels', default="[64,64,32,32,1]") 
    parser.add_argument('--gen_strides', default="[1,2,2,2,2]")
    parser.add_argument('--gen_paddings', default="[1,1,1,1,1]")  
    parser.add_argument('--gen_activation', default='ReLU')
    parser.add_argument('--likelihood', default='gaussian', choices=['gaussian', 'cb'])
    
    parser.add_argument('--num_clusters', default=20, type=int)
    ## training config
    parser.add_argument('--sample_size', default=20, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    return parser.parse_args()

if __name__== "__main__":
    args = parse_args()
    main(args)