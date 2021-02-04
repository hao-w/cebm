import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import time
"""
training procedure for standard BI-GAN.
"""
class Train_procedure():
    def __init__(self, optimizerD, optimizerGE, disc, gen, train_data, num_epochs, batch_size, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizerD = optimizerD
        self.optimizerGE = optimizerGE
        self.enc = enc
        self.disc = disc
        self.gen = gen
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.save_version = save_version
        self.real_label = torch.ones(self.batch_size).cuda().to(device).float()
        self.fake_label = torch.zeros(self.batch_size).cuda().to(device).float()
        self.bceloss = nn.BCELoss()
    
    def reset_grad(self):
        self.enc.zero_grad()
        self.disc.zero_grad()
        self.gen.zero_grad()
        
    def train(self):
#         torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.num_epochs):
            time_start = time.time()
            metrics = {'lossD': 0.0, 'lossG': 0.0}
            for b, (images, _) in enumerate(self.train_data): 
                total = 150
                # zero mean, std = 0.1 * (self.args.num_epochs - epoch) / self.args.num_epochs
                noise_std = 0.1 * (total - epoch) / float(total)
                noise1 = torch.randn_like(images, device=self.device, requires_grad=False) * noise_std
                noise2 = torch.randn_like(images, device=self.device, requires_grad=False) * noise_std
                
                real_images = images.cuda().to(self.device)
                # generate fake images
                fake_z, fake_images = self.gen.forward(real_images.shape[0])
                # encode real images
                real_z = self.enc.forward(real_images)
                # discrinimate real and fake examples
                pred_fake = self.disc.binary_pred(fake_images + noise1, fake_z)
                pred_real = self.disc.binary_pred(real_images + noise2, real_z)

                # update GE
                self.optimizerGE.zero_grad()
                if pred_fake.shape[0] == self.real_label.shape[0]:
                    loss_ge = self.bceloss(pred_fake, self.real_label) + self.bceloss(pred_real, self.fake_label)
                else:
                    loss_ge = self.bceloss(pred_fake, self.real_label[:pred_fake.shape[0]]) + self.bceloss(pred_real, self.fake_label[:pred_fake.shape[0]])
                    
                loss_ge.backward()
                self.optimizerGE.step()
                
                # update D
                self.optimizerD.zero_grad()    
                # discrinimate real and fake examples
                pred_fake = self.disc.binary_pred(fake_images.detach() + noise2, fake_z.detach())
                pred_real = self.disc.binary_pred(real_images + noise1, real_z.detach())
                if pred_real.shape[0] == self.real_label.shape[0]:
                    loss_d = self.bceloss(pred_real, self.real_label) + self.bceloss(pred_fake, self.fake_label)
                else:
                    loss_d = self.bceloss(pred_real, self.real_label[:pred_real.shape[0]]) + self.bceloss(pred_fake, self.fake_label[:pred_real.shape[0]])
                
                loss_d.backward()
                self.optimizerD.step()
                self.reset_grad()

                

                
                metrics['lossD'] += loss_d.detach()
                metrics['lossG'] += loss_ge.detach()
#                 metrics['Dpred_real'] += Dpred_real
#                 metrics['Dpred_fake'] += Dpred_fake
#                 metrics['Gpred_fake'] += Gpred_fake
            self.save_checkpoints()
            self.logging(metrics=metrics, N=b+1, epoch=epoch)
            time_end = time.time()
            print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))

#     def optimize_disc(self, images, fake_images):
#         pred_real = self.disc.binary_pred(images)
#         if pred_real.shape[0] == self.real_label.shape[0]:
#             errD_real = self.bceloss(pred_real, self.real_label)
#             pred_fake = self.disc.binary_pred(fake_images.detach())
#             errD_fake = self.bceloss(pred_fake, self.fake_label)
#         else:
#             errD_real = self.bceloss(pred_real, self.real_label[:pred_real.shape[0]])
#             pred_fake = self.disc.binary_pred(fake_images.detach())
#             errD_fake = self.bceloss(pred_fake, self.fake_label[:pred_fake.shape[0]])
#         return errD_real + errD_fake, pred_real.mean().item(), pred_fake.mean().item()
    
#     def optimize_gen_enc(self):
#         fake_z, fake_images = self.gen.forward(self.batch_size)
#         pred_fake = self.disc.binary_pred(fake_images, fake_z)
#         if pred_fake.shape[0] == self.real_label.shape[0]:
#             errG_fake = self.bceloss(pred_fake, self.real_label)
#         else:
#             errG_fake = self.bceloss(pred_fake, self.real_label[:pred_fake.shape[0]])
#         return errG_fake, pred_fake.mean().item(), fake_images
    
    def logging(self, metrics, N, epoch):
        if epoch == 0:
            log_file = open('results/log-' + self.save_version + '.txt', 'w+')
        else:
            log_file = open('results/log-' + self.save_version + '.txt', 'a+')
        metrics_print = ",  ".join(['%s=%.3f' % (k, v / N) for k, v in metrics.items()])
        print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
        log_file.close()
        
    def save_checkpoints(self):
        checkpoint_dict  = {
            'enc_state_dict': self.enc.state_dict(),
            'disc_state_dict': self.disc.state_dict(),
            'gen_state_dict': self.gen.state_dict(),
            'gen_prior_mu' : self.gen.prior_mu,
            'gen_prior_log_sigma' : self.gen.prior_log_sigma}
        torch.save(checkpoint_dict, "weights/cp-%s" % self.save_version)

        
if __name__ == "__main__":
    import torch
    import argparse
    from sebm.data import load_data
    from sebm.models import Generator_GMM, Discriminator_BIGAN, Encoder_BIGAN
    from util import set_seed
    
    parser = argparse.ArgumentParser('BIGAN')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet', 'celeba', 'flowers102', 'fashionmnist'])    
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    ## arch config
    parser.add_argument('--arch', default='simplenet', choices=['simplenet'])
#     parser.add_argument('--disc_channels', default="[32,64,64,128]")
#     parser.add_argument('--disc_kernels', default="[4, 4, 4, 4]")
#     parser.add_argument('--disc_strides', default="[2, 2, 2, 2]")
#     parser.add_argument('--disc_paddings', default="[1, 1, 1, 1]")
    parser.add_argument('--disc_channels', default="[64, 64, 32, 32]")
    parser.add_argument('--disc_kernels', default="[3, 4, 4, 4]")
    parser.add_argument('--disc_strides', default="[1, 2, 2, 2]")
    parser.add_argument('--disc_paddings', default="[1, 1, 1, 1]")

    
    parser.add_argument('--gen_kernels', default="[4, 4, 3, 4, 4]")
    parser.add_argument('--gen_channels', default="[64, 64, 32, 32, 1]")    
    parser.add_argument('--gen_strides', default="[1, 2, 2, 2, 2]")
    parser.add_argument('--gen_paddings', default="[1, 1, 1, 1, 1]")
    
    parser.add_argument('--enc_channels', default="[64,64,32,32]")
    parser.add_argument('--enc_kernels', default="[3, 4, 4, 4]")
    parser.add_argument('--enc_strides', default="[1, 2, 2, 2]")
    parser.add_argument('--enc_paddings', default="[1, 1, 1, 1]")
    
    parser.add_argument('--cnn_output_dim', default=288, type=int)
    
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=128, type=int)
    
    parser.add_argument('--disc_activation', default='LeakyReLU')
    parser.add_argument('--gen_activation', default='ReLU')

    parser.add_argument('--leak', default=0.2, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    ## training config
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--gmm_components', default=50, type=int)
#     parser.add_argument('--reparameterized', default=True, )
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:%d' % args.device)
    save_version = 'bigan_gmm-d=%s-seed=%d-lr=%s-zd=%d-disc_act=%s-gen_act=%s-arch=%s' % (args.dataset, args.seed, args.lr, args.latent_dim, args.disc_activation, args.gen_activation, args.arch)
    print('Experiment with ' + save_version)
    print('Loading dataset=%s...' % args.dataset)
    train_data, img_dims = load_data(args.dataset, args.data_dir, args.batch_size, train=True, normalize=True)
    (input_channels, im_height, im_width) = img_dims 
    print('Initialize BIGAN...')
    reparameterized = True
    enc = Encoder_BIGAN(arch=args.arch,
                        reparameterized=reparameterized,
                        im_height=im_height, 
                        im_width=im_width, 
                        input_channels=input_channels, 
                        channels=eval(args.enc_channels), 
                        kernels=eval(args.enc_kernels), 
                        strides=eval(args.enc_strides), 
                        paddings=eval(args.enc_paddings), 
                        hidden_dim=eval(args.hidden_dim),
                        latent_dim=args.latent_dim,
                        activation=args.disc_activation,
                        leak=args.leak,
                        last_act=False,
                        batchnorm=True)  

    gen = Generator_GMM(arch=args.arch,
                        learn_prior=True,
                        K=args.gmm_components,
                        device=device,
                        im_height=1, 
                        im_width=1, 
                        input_channels=args.latent_dim, 
                        channels=eval(args.gen_channels), 
                        kernels=eval(args.gen_kernels), 
                        strides=eval(args.gen_strides), 
                        paddings=eval(args.gen_paddings), 
                        activation=args.gen_activation,
                        leak=args.leak,
                        batchnorm=True)
    
    disc = Discriminator_BIGAN(arch=args.arch,
                               latent_dim=args.latent_dim,
                               cnn_output_dim=args.cnn_output_dim,
                               hidden_dim=eval(args.hidden_dim),
                                im_height=im_height, 
                                im_width=im_width, 
                                input_channels=input_channels, 
                                channels=eval(args.disc_channels), 
                                kernels=eval(args.disc_kernels), 
                                strides=eval(args.disc_strides), 
                                paddings=eval(args.disc_paddings), 
                                activation=args.disc_activation,
                                leak=args.leak,
                                last_act=True,
                                batchnorm=True,
                                dropout=args.dropout)    
    enc = enc.cuda().to(device)
    gen = gen.cuda().to(device)
    disc = disc.cuda().to(device)
    beta1 = 0.5
    optimizerD = getattr(torch.optim, args.optimizer)(list(disc.parameters()), lr=args.lr,betas=(beta1, 0.999))
    optimizerGE = getattr(torch.optim, args.optimizer)(list(gen.parameters())+list(enc.parameters()), lr=args.lr,betas=(beta1, 0.999))
    print('Start training...')
    trainer = Train_procedure(optimizerD, optimizerGE, disc, gen, train_data, args.num_epochs, args.batch_size, device, save_version)
    trainer.train()
