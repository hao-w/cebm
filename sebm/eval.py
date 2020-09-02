import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sebm.data import load_data
from sebm.cebm_sgld import SGLD_sampler
from sebm.gaussian_params import nats_to_params
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

class Evaluator_EBM():
    """
    evaluator for a conjugate-EBM
    """
    def __init__(self, ebm, device, dataset, data_dir, data_noise_std):
        super(self.__class__, self).__init__()
        self.ebm = ebm
        self.device = device
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_noise_std = data_noise_std
        if dataset == 'mnist' or dataset =='fashionmnist':
            self.input_channels, self.im_height, self.im_width = 1, 28, 28
        else:
            self.input_channels, self.im_height, self.im_width = 3, 32, 32
            
    def uncond_sampling(self, batch_size, sgld_steps, sgld_lr, sgld_noise_std, grad_clipping=False):
        sgld_sampler = SGLD_sampler(device=self.device,
                                    input_channels=self.input_channels,
                                    noise_std=sgld_noise_std,
                                    lr=sgld_lr,
                                    pixel_size=self.im_height,
                                    buffer_size=None,
                                    buffer_percent=None,
                                    buffer_init=False,
                                    buffer_dup_allowed=False,
                                    grad_clipping=grad_clipping)
        images_ebm, list_samples = sgld_sampler.sample(ebm=self.ebm, 
                                         batch_size=batch_size, 
                                         num_steps=sgld_steps,
                                         pcd=False)
        return images_ebm, list_samples

    def plot_samples(self, images, fs=10, save_name=None):
        test_batch_size = len(images)
        images = images.squeeze().cpu().detach()
        images = torch.clamp(images, min=-1, max=1)
        images = images * 0.5 + 0.5
        gs = gridspec.GridSpec(int(test_batch_size/10), 10)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
        fig = plt.figure(figsize=(fs, fs*int(test_batch_size/10)/ 10))
        for i in range(test_batch_size):
            ax = fig.add_subplot(gs[int(i/10), i%10])
            try:
                ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(images[i], (1,2,0)), vmin=0, vmax=1.0)
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
        if save_name is not None:
            plt.savefig('samples/' + save_name + '_samples.png', dpi=300)
            plt.close()
            
    def extract_energy(self, dataset, train):
        print('Loading dataset=%s...' % self.dataset)
        test_data, img_dims = load_data(dataset, self.data_dir, 1000, train=train)
        Es = []
        for (images, _) in test_data:
            images = images.cuda().to(self.device)
            images = images + self.data_noise_std * torch.randn_like(images)
            energy = self.ebm.energy(images)
            Es.append(energy.cpu().detach().numpy())
        Es =  np.concatenate(Es, 0)
        return Es

    def extract_features(self, train, sample_size=None):
        print('Loading dataset=%s...' % self.dataset)
        test_data, img_dims = load_data(self.dataset, self.data_dir, 1000, train=train)
        zs = []
        ys = []
        for (images, labels) in test_data:
            images = images.cuda().to(self.device)
            images = images + self.data_noise_std * torch.randn_like(images)
            neural_ss1, neural_ss2 = self.ebm.forward(images)
            if sample_size is not None:
                latents, _ = self.ebm.sample_posterior(sample_size, neural_ss1, neural_ss2)
                zs.append(latents.squeeze().cpu().detach().numpy())
            else:
                mean, sigma = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
                zs.append(mean.squeeze().cpu().detach().numpy())
            ys.append(labels)
        zs = np.concatenate(zs, 0)
        ys = np.concatenate(ys, 0)
        return zs, ys
    
    def compute_tsne(self):
        print('extract mean feature of testing data using EBM..')
        zs, ys = self.extract_features(self.data_noise_std, train=False)
        print('transform latent to 2D tsne features..')  
        zs2 = TSNE().fit_transform(zs)
        return zs2, ys

    def plot_tsne(self, zs2, ys, fs=8, ms=10, save_name=None):
        print('plotting tsne figure..')
        fig = plt.figure(figsize=(fs,fs))
        ax = fig.add_subplot(111)
        num_classes = len(np.unique(ys))
        colors = []
        for k in range(num_classes):
            m = (ys == k)
            p = ax.scatter(zs2[m, 0], zs2[m, 1], label='y=%d' % k, alpha=0.5, s=ms)
            colors.append(p.get_facecolor())
        ax.legend()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        if save_name is not None:
            plt.savefig(save_name + '_tsne.png', dpi=300) 
            
    def plot_hist(self, energies, bins=100, save_name=None):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        for name, energy in energies.items():
            _ = ax.hist(energy, bins=100, label=key, alpha=.3, density=True)
        ax.set_title('Train on %s' % self.dataset)
        ax.set_xlabel('E(x)')
        ax.legend(fontsize=14)
        if save_name is not None:
            plt.savefig('hist_'+ save_name + '.png', dpi=300)            
         
        
    def train_logistic_classifier(self):
        print('extract mean feature of training data using EBM..')
        zs, ys = self.extract_features(train=True)
        print('start to train lr classifier..')
        lr = LogisticRegression(random_state=0, multi_class='multinomial', solver='saga').fit(zs, ys)
        print('extract mean feature of testing data using EBM..')
        zs_test, ys_test = self.extract_features(train=False)
        print('testing lr classifier..')
        accu = lr.score(zs_test, ys_test)
        print('mean accuray=%.4f' % accu)

#     def train_knn_classifier(self):
#         print('extract mean feature of training data using EBM..')
#         zs, ys = self.extract_features(self.data_noise_std, train=True)
#         num_classes = len(np.unique(ys))
#         print('start to train knn classifier..')
#         knn = NearestNeighbors(n_neighbors=num_classes).fit(zs, ys)
#         print('extract mean feature of testing data using EBM..')
#         zs_test, ys_test = self.extract_features(self.data_noise_std, train=False)
#         print('testing knn classifier..')
#         accu = knn.score(zs_test, ys_test)        
#         print('mean accuray=%.4f' % accu)
        
    def plot_post_under_prior(self, train, density=False, save_name=None):
        print('extract mean feature of data using EBM..')
        zs, ys = self.extract_features(train=train)
        prior_mu, prior_sigma = nats_to_params(self.ebm.prior_nat1, self.ebm.prior_nat2)
        prior_dist = Normal(prior_mu, prior_sigma)
        zs = torch.Tensor(zs).cuda().to(self.device)
        post_log_prob = prior_dist.log_prob(zs).sum(-1).cpu().detach().numpy()
        prior_log_prob = prior_dist.log_prob(prior_dist.sample((len(ys), ))).sum(-1).cpu().detach().numpy()
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.hist(post_log_prob, bins=100, label='q(z|x)', alpha=.3, density=density)
        ax.set_title('Train on %s' % self.dataset, fontsize=14)
        _ = ax.hist(prior_log_prob, bins=100, label='p(z)' , alpha=.3, density=density)
        ax.legend(fontsize=14)
        if save_name is not None:
            plt.savefig('hist_'+ save_name + '.png', dpi=300)            
                 
class Evaluator_VAE():
    """
    Evaluator for a vanilla VAE
    """
    def __init__(self, enc, dec, device, dataset, data_dir):
            super(self.__class__, self).__init__()
            self.enc = enc
            self.dec = dec
            self.device = device
            self.dataset = dataset
            self.data_dir = data_dir
            if dataset == 'mnist' or dataset =='fashionmnist':
                self.input_channels, self.im_height, self.im_width = 1, 28, 28
            else:
                self.input_channels, self.im_height, self.im_width = 3, 32, 32 
                
    def test_one_batch(self, batch_size):
        test_data, _ = load_data(self.dataset, self.data_dir, batch_size=batch_size, train=False, normalize=False)
        for (images, _) in test_data:
            break
        images = images.cuda().to(self.device)
        latents, q_log_prob = self.enc.forward(images)
        recons, ll, p_log_prob = self.dec.forward(latents, images)
        return images, recons.view(batch_size, self.im_height, self.im_width)
        
    def plot_samples(self, images, recons, fs=10, data_name=None):
        test_batch_size = len(images)
        images = images.squeeze().cpu().detach()
        recons = recons.squeeze().cpu().detach()
        gs = gridspec.GridSpec(int(test_batch_size/10)*2, 10)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.2, hspace=0.2)
        fig = plt.figure(figsize=(fs, fs*int(test_batch_size/10)*2 / 10))
        for i in range(test_batch_size):
            ax = fig.add_subplot(gs[int(i/10)*2, i%10])
            try:
                ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(images[i], (1,2,0)), vmin=0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax = fig.add_subplot(gs[int(i/10)*2+1, i%10])
            try:
                ax.imshow(recons[i], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(recons[i], (1,2,0)), vmin=0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
        if data_name is not None:
            plt.savefig(data_name + '_samples.png', dpi=300)
            
    def extract_features(self, train):
        zs = []
        ys = []
        train_data, img_dims = load_data(self.dataset, self.data_dir, 1000, train=train)   
        for b, (images, labels) in enumerate(train_data):
            images = images.cuda().to(self.device)
            mu, _ = self.enc.enc_net(images)
            zs.append(mu.cpu().detach().numpy())
            ys.append(labels)
        zs = np.concatenate(zs, 0)
        ys = np.concatenate(ys, 0)
        return zs, ys   
    
    def compute_tsne(self):
        print('extract mean feature of testing data using EBM..')
        zs, ys = self.extract_features(train=False)
        print('transform latent to 2D tsne features..')  
        zs2 = TSNE().fit_transform(zs)
        return zs2, ys

    def plot_tsne(self, zs2, ys, fs=8, ms=10, save_name=None):
        print('plotting tsne figure..')
        fig = plt.figure(figsize=(fs,fs))
        ax = fig.add_subplot(111)
        num_classes = len(np.unique(ys))
        colors = []
        for k in range(num_classes):
            m = (ys == k)
            p = ax.scatter(zs2[m, 0], zs2[m, 1], label='y=%d' % k, alpha=0.5, s=ms)
            colors.append(p.get_facecolor())
        ax.legend()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        if save_name is not None:
            plt.savefig(save_name + '_tsne.png', dpi=300) 
            
        
    def train_logistic_classifier(self):
        print('extract mean feature of training data using EBM..')
        zs, ys = self.extract_features(train=True)
        print('start to train lr classifier..')
        lr = LogisticRegression(random_state=0, multi_class='multinomial', solver='saga').fit(zs, ys)
        print('extract mean feature of testing data using EBM..')
        zs_test, ys_test = self.extract_features(train=False)
        print('testing lr classifier..')
        accu = lr.score(zs_test, ys_test)
        print('mean accuray=%.4f' % accu)

#     def train_knn_classifier(self):
#         print('extract mean feature of training data using EBM..')
#         zs, ys = self.extract_features(train=True)
#         num_classes = len(np.unique(ys))
#         print('start to train knn classifier..')
#         knn = NearestNeighbors(n_neighbors=num_classes).fit(zs, ys)
#         print('extract mean feature of testing data using EBM..')
#         zs_test, ys_test = self.extract_features(train=False)
#         print('testing knn classifier..')
#         accu = knn.score(zs_test, ys_test)        
#         print('mean accuray=%.4f' % accu)

    def plot_post_under_prior(self, train, density=False, save_name=None):
        print('extract mean feature of data using EBM..')
        zs, ys = self.extract_features(train=train)
        prior_dist = Normal(self.dec.prior_mu, self.dec.prior_sigma)
        zs = torch.Tensor(zs).cuda().to(self.device)
        post_log_prob = prior_dist.log_prob(zs).sum(-1).cpu().detach().numpy()
        prior_log_prob = prior_dist.log_prob(prior_dist.sample((len(ys), ))).sum(-1).cpu().detach().numpy()
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.hist(post_log_prob, bins=100, label='q(z|x)', alpha=.3, density=density)
        ax.set_title('Train on %s' % self.dataset, fontsize=14)
        _ = ax.hist(prior_log_prob, bins=100, label='p(z)' , alpha=.3, density=density)
        ax.legend(fontsize=14)
        if save_name is not None:
            plt.savefig('hist_'+ save_name + '.png', dpi=300)            
                 