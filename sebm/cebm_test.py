import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import time
"""
training procedure for conjugate EBM with latent variable z, using learned proposal for sampling from model distribution.
"""

        
class Train_procedure():
    def __init__(self, optimizer, ebm, sgld_sampler, sgld_num_steps, data_noise_std, train_data, num_epochs, sample_size, regularize_factor, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizer = optimizer
        self.ebm = ebm
        self.sgld_sampler = sgld_sampler
        self.sgld_num_steps = sgld_num_steps
        self.data_noise_std = data_noise_std
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.sample_size = sample_size
#         self.batch_size = batch_size
        self.reg_alpha = regularize_factor
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
                trace = self.jkl(images)
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
                print('pass!')
            self.save_checkpoints()
            self.logging(metrics=metrics, N=b+1, epoch=epoch)
            time_end = time.time()
            print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))

    def jkl(self, images_data):
        """
        objective: - ull_term + kl_term
        ull_term : unnormalized likelihood term E[log \pi(x | z)]
        kl_term : KL(q(z | x) || p(z))
        """ 
        trace = dict()
        batch_size, C, pixels_size, _ = images_data.shape
        neural_ss1, neural_ss2 = self.ebm.forward(images_data)
        latents, log_posterior = self.ebm.latent_posterior(self.sample_size, neural_ss1, neural_ss2)
        log_prior = self.ebm.latent_prior(self.sample_size, batch_size, latents=latents)
        kl_term = (log_posterior - log_prior).mean()
        log_f_data = self.ebm.log_factor(self.sample_size, neural_ss1, neural_ss2, latents)
        
        images_neg = # TODO implement negative sampling 
        neural_ss1_neg, neural_ss2_neg = self.ebm.forward(images_neg)
        log_f_neg = self.ebm.log_factor(self.sample_size, neural_ss1_neg, neural_ss2_neg, latents)
        ull_term = (log_f_data - log_f_neg).mean()
        trace['loss'] = - ull_term + kl_term
        return trace
    
    
    def logging(self, metrics, N, epoch):
        if epoch == 0:
            log_file = open('results/log-' + self.save_version + '.txt', 'w+')
        else:
            log_file = open('results/log-' + self.save_version + '.txt', 'a+')
        metrics_print = ",  ".join(['%s=%.3e' % (k, v / N) for k, v in metrics.items()])
        print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
        log_file.close()
        
    def save_checkpoints(self):
        checkpoint_dict  = {
            'model_state_dict': self.ebm.state_dict()
#             'replay_buffer': self.sgld_sampler.buffer
            }
        torch.save(checkpoint_dict, "weights/checkpoint-%s" % self.save_version)


    
if __name__ == "__main__":
    import torch
    import argparse
    from sebm.data import load_data  
    from sebm.models import CEBM_test
    from util import set_seed
    parser = argparse.ArgumentParser('Conjugate EBM')
    parser.add_argument('--ss', default='2', choices=['1', '2'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'cifar100', 'celeba', 'flowers102'])
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--sample_size', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--data_noise_std', default=1e-2, type=float)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--optimize_priors', default=False, type=bool)
    ## arch config
    parser.add_argument('--arch', default='simplenet', choices=['simplenet', 'wresnet'])
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
    ## regularization config
    parser.add_argument('--regularize_factor', default=1e-3, type=float)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:%d' % args.device)
    save_version = 'cebm-ss=%s-dataset=%s-seed=%d-lr=%s-latentdim=%d-data_noise_std=%s-reg_alpha=%s-act=%s-arch=%s' % (args.ss, args.dataset, args.seed, args.lr, args.latent_dim, args.data_noise_std, args.regularize_factor, args.activation, args.arch)
    print('Experiment with ' + save_version)
    print('Loading dataset=%s...' % args.dataset)
    train_data, img_dims = load_data(args.dataset, args.data_dir, args.batch_size, train=True)
    (input_channels, im_height, im_width) = img_dims  
    model = eval('CEBM_test')
    print('Initialize Model=%s...' % model.__name__)    
    ebm = model(arch=args.arch,
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
    ebm = ebm.cuda().to(device)
    optimizer = getattr(torch.optim, args.optimizer)(list(ebm.parameters()), lr=args.lr)
    print('Start training...')
    trainer = Train_procedure(optimizer, ebm, args.data_noise_std, train_data, args.num_epochs, args.sample_size, args.regularize_factor, device, save_version)
    trainer.train()
