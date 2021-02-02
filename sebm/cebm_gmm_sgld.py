import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import time
from sebm.models import CEBM_GMM_2ss
"""
training procedure for conjugate EBM with latent variable z, using SGLD for sampling from model distribution.
"""
        
class Train_procedure():
    def __init__(self, optimizer, ebm, sgld_sampler, sgld_num_steps, data_noise_std, train_data, num_epochs, regularize_factor, lr, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizer = optimizer
        self.ebm = ebm
        self.sgld_sampler = sgld_sampler
        self.sgld_num_steps = sgld_num_steps
        self.data_noise_std = data_noise_std
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.reg_alpha = regularize_factor
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
            time_end = time.time()
            self.logging(metrics=metrics, N=b+1, epoch=epoch, ts=time_end-time_start)
            time_end = time.time()
            print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))
        torch.save(sgld_sampler.buffer, 'weights/buffer-%s' % self.save_version)

    def pcd(self, images_data):
        """
        we acquire samples from ebm using stochastic gradient langevin dynamics
        """ 
        trace = dict()
        batch_size, C, pixels_size, _ = images_data.shape
        energy_data = self.ebm.energy(images_data)
        images_ebm = self.sgld_sampler.sample(ebm, batch_size, self.sgld_num_steps, pcd=True)
        energy_ebm = ebm.energy(images_ebm)
        trace['loss'] = (energy_data - energy_ebm).mean() + self.reg_alpha * ((energy_data**2).mean() +(energy_ebm**2).mean())
        trace['energy_data'] = energy_data.detach().mean()
        trace['energy_ebm'] = energy_ebm.detach().mean()
        return trace
    
    
    def logging(self, metrics, N, epoch, ts):
        if epoch == 0:
            log_file = open('results/log-' + self.save_version + '.txt', 'w+')
        else:
            log_file = open('results/log-' + self.save_version + '.txt', 'a+')
        metrics_print = ",  ".join(['%s=%.3e' % (k, v / N) for k, v in metrics.items()])
        print("(%ds)Epoch=%d, " % (ts, epoch+1) + metrics_print, file=log_file)
        log_file.close()        
    def save_checkpoints(self):
        checkpoint_dict  = {
            'model_state_dict': self.ebm.state_dict(),
            'prior_mu' : self.ebm.prior_mu,
            'prior_log_sigma' : self.ebm.prior_log_sigma
            #'replay_buffer' : self.sgld_sampler.buffer
            }
        torch.save(checkpoint_dict, "weights/cp-%s" % self.save_version)

if __name__ == "__main__":
    import torch
    import argparse
    from sebm.data import load_data
    from sebm.models import CEBM_GMM_2ss
    from sebm.sgld import SGLD_sampler
    from util import set_seed
    parser = argparse.ArgumentParser('Conjugate EBM')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet', 'celeba', 'flowers102', 'fashionmnist'])
    parser.add_argument('--data_dir', default=None, type=str)
#     parser.add_argument('--sample_size', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_noise_std', default=1e-2, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--optimize_priors', default=False, action='store_true')
    parser.add_argument('--num_clusters', default=10, type=int)
    ## arch config
    parser.add_argument('--arch', default='simplenet', choices=['simplenet', 'simplenet2'])
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--channels', default="[64, 64, 32, 32]")
    parser.add_argument('--kernels', default="[3, 4, 4, 4]")
    parser.add_argument('--strides', default="[1, 2, 2, 2]")
    parser.add_argument('--paddings', default="[1, 1, 1, 1]")
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--activation', default='Swish')
    parser.add_argument('--leak', default=0.01, type=float)
    ## training config
    parser.add_argument('--num_epochs', default=200, type=int)
    ## sgld sampler config
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--buffer_percent', default=0.95, type=float)
    parser.add_argument('--buffer_init', default=False, action='store_true')
    parser.add_argument('--buffer_dup_allowed', default=False, action='store_true')
    parser.add_argument('--sgld_noise_std', default=7.5e-3, type=float)
    parser.add_argument('--sgld_lr', default=1.0, type=float)
    parser.add_argument('--sgld_num_steps', default=50, type=int)
    parser.add_argument('--grad_clipping', default=False, action='store_true')
    ## regularization config
    parser.add_argument('--regularize_factor', default=1e-3, type=float)
#     parser.add_argument('--heldout_class', default=-1, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1])

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:%d' % torch.cuda.current_device())
    save_version = 'cebm_gmm_k=%d-d=%s-seed=%d-lr=%s-zd=%d-d_ns=%s-sgld-ns=%s-lr=%s-steps=%s-reg=%s-act=%s-arch=%s' % (args.num_clusters, args.dataset, args.seed, args.lr, args.latent_dim, args.data_noise_std, args.sgld_noise_std, args.sgld_lr, args.sgld_num_steps, args.regularize_factor, args.activation, args.arch)
    print('Experiment with ' + save_version)
    print('Loading dataset=%s...' % args.dataset)
    train_data, img_dims = load_data(args.dataset, args.data_dir, args.batch_size, train=True, normalize=True)
    (input_channels, im_height, im_width) = img_dims  
    if args.arch == 'simplenet' or args.arch == 'simplenet2':
        ebm = CEBM_GMM_2ss(K=args.num_clusters,
                        arch=args.arch,
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
        
    ebm = ebm.cuda().to(device)
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
    trainer = Train_procedure(optimizer, ebm, sgld_sampler, args.sgld_num_steps, args.data_noise_std, train_data, args.num_epochs, args.regularize_factor, args.lr, device, save_version)
    trainer.train()
