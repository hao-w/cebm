import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import time
from sebm.models import CEBM_2ss
"""
training procedure for conjugate EBM with latent variable z, using SGLD for sampling from model distribution.
"""
        
class Train_procedure():
    def __init__(self, optimizer, ebm, sgld_sampler, sgld_num_steps, data_noise_std, train_data, num_epochs, regularize_factor, sample_size, lr, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizer = optimizer
        self.ebm = ebm
        self.sgld_sampler = sgld_sampler
        self.sgld_num_steps = sgld_num_steps
        self.data_noise_std = data_noise_std
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.reg_alpha = regularize_factor
        self.sample_size = sample_size
        self.lr = lr
        self.device = device
        self.save_version = save_version
        
    def train(self):
        for epoch in range(self.num_epochs):
            time_start = time.time()
            metrics = dict()
            for b, (images, _) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                images = images.cuda().to(self.device)
                images = images + self.data_noise_std * torch.randn_like(images)
                trace = self.pcd(images)
                if trace['loss'].abs().item() > 1e+8:
                    print('Model is diverging, will terminate training..')
                    exit()
                trace['loss'].backward()
                self.optimizer.step()
                for key in trace.keys():
                    if key not in metrics:
                        metrics[key] = trace[key].detach()
                    else:
                        metrics[key] += trace[key].detach() 
            self.save_checkpoints()
            self.logging(metrics=metrics, N=b+1, epoch=epoch)
            time_end = time.time()
            print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))
        torch.save(sgld_sampler.buffer, 'weights/buffer-%s' % self.save_version)

    def pcd(self, data_images):
        """
        we acquire samples from ebm using stochastic gradient langevin dynamics
        """ 
        trace = dict()
        batch_size, C, pixels_size, _ = data_images.shape
        # extra term for maximizing the likelihood
        data_log_factor = self.ebm.log_factor_posterior(data_images, self.sample_size)
        # standard CD objective for marginal
        energy_data = self.ebm.energy(data_images)
        images_ebm = self.sgld_sampler.sample(ebm, batch_size, self.sgld_num_steps, pcd=True)
        energy_ebm = ebm.energy(images_ebm)
        trace['loss'] = (energy_data - energy_ebm).mean() + self.reg_alpha * ((energy_data**2).mean() + (energy_ebm**2).mean()) - data_log_factor.mean()
        trace['E_data'] = energy_data.detach().mean()
        trace['E_ebm'] = energy_ebm.detach().mean()
        trace['ll'] = data_log_factor.detach().mean()
        return trace
    
    
    def logging(self, metrics, N, epoch):
        if epoch == 0:
            log_file = open('results/log-' + self.save_version + '.txt', 'w+')
        else:
            log_file = open('results/log-' + self.save_version + '.txt', 'a+')
        metrics_print = ",  ".join(['%s=%.2e' % (k, v / N) for k, v in metrics.items()])
        print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
        log_file.close()
        
    def save_checkpoints(self):
        checkpoint_dict  = {
            'model_state_dict': self.ebm.state_dict()
            }
        torch.save(checkpoint_dict, "weights/cp-%s" % self.save_version)


def init_cebm(arch, ss, latent_dim, optimize_priors, device, **kwargs):
    model = eval('CEBM_%sss' % ss)
    print('Initialize Model=%s...' % model.__name__)
    ebm = model(arch, latent_dim, optimize_priors, device, **kwargs)
    return ebm

if __name__ == "__main__":
    import torch
    import argparse
    from sebm.data import load_data, load_mnist_heldout
    from sebm.models import CEBM_2ss
    from sebm.sgld import SGLD_sampler
    from util import set_seed
    parser = argparse.ArgumentParser('Conjugate EBM')
    parser.add_argument('--ss', default='2', choices=['1', '2'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--sample_size', default=10, type=int)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet', 'celeba', 'fashionmnist'])
    parser.add_argument('--data_dir', default='../../sebm_data/', type=str)
#     parser.add_argument('--sample_size', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_noise_std', default=0.0, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--optimize_priors', default=False, type=bool)
    ## arch config
    parser.add_argument('--arch', default='simplenet2', choices=['simplenet2'])
    parser.add_argument('--channels', default="[64,128,256,512]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--activation', default='Swish')
    parser.add_argument('--leak', default=0.1, type=float)
    ## training config
    parser.add_argument('--num_epochs', default=150, type=int)
    ## sgld sampler config
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--buffer_percent', default=0.95, type=float)
    parser.add_argument('--buffer_init', default=False, action='store_true')
    parser.add_argument('--buffer_dup_allowed', default=False, action='store_true')
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_lr', default=2.0, type=float)
    parser.add_argument('--sgld_num_steps', default=50, type=int)
    parser.add_argument('--grad_clipping', default=False, action='store_true')
    ## regularization config
    parser.add_argument('--regularize_factor', default=1e-1, type=float)
    parser.add_argument('--dropout', default=None, type=float)
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:%d' % args.device)
    save_version = 'cebm_joint-d=%s-seed=%d-lr=%s-zd=%d-d_ns=%s-sgld-ns=%s-lr=%s-steps=%s-reg=%s-act=%s' % (args.dataset, args.seed, args.lr, args.latent_dim, args.data_noise_std, args.sgld_noise_std, args.sgld_lr, args.sgld_num_steps, args.regularize_factor, args.activation)
    print('Experiment with ' + save_version)
    print('Loading dataset=%s...' % args.dataset)
    train_data, img_dims = load_data(args.dataset, args.data_dir, args.batch_size, train=True)
    (input_channels, im_height, im_width) = img_dims  
        
    if args.arch == 'simplenet' or args.arch == 'simplenet2':
        ebm = init_cebm(arch=args.arch,
                        ss=args.ss,
                        optimize_priors=args.optimize_priors,
                        device=device,
                        im_height=im_height, 
                        im_width=im_width, 
                        input_channels=input_channels, 
                        channels=eval(args.channels), 
                        kernels=eval(args.kernels), 
                        strides=eval(args.strides), 
                        paddings=eval(args.paddings), 
                        hidden_dim=eval(args.hidden_dim),
                        latent_dim=args.latent_dim,
                        activation=args.activation,
                        leak=args.leak)
    else:
        raise NotImplementError
        
    ebm = ebm.to(device)
    optimizer = getattr(torch.optim, args.optimizer)(list(ebm.parameters()), lr=args.lr)
    print('Initialize sgld sampler...')
    sgld_sampler = SGLD_sampler(device=device,
                                input_channels=input_channels,
                                noise_std=args.sgld_noise_std,
                                lr=args.sgld_lr,
                                pixel_size=im_height,
                                buffer_size=args.buffer_size,
                                buffer_percent=args.buffer_percent,
                                buffer_init=args.buffer_init,
                                buffer_dup_allowed=args.buffer_dup_allowed,
                                grad_clipping=args.grad_clipping)
    
    print('Start training...')
    trainer = Train_procedure(optimizer, ebm, sgld_sampler, args.sgld_num_steps, args.data_noise_std, train_data, args.num_epochs, args.regularize_factor, args.sample_size, args.lr, device, save_version)
    trainer.train()
