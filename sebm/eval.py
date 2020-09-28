import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sebm.data import load_data, load_data_as_array
from sebm.cebm_sgld import SGLD_sampler
from sebm.gaussian_params import nats_to_params
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier

def train_logistic_classifier(evaluator, train_data=None):
    if train_data is None:
        print('extract mean feature of training data..')
        zs, ys = evaluator.extract_features(train=True)
    else:
        zs, ys = evaluator.extract_features_fewshots(train_data)
    print('start to train lr classifier..')
    lr = LogisticRegression(random_state=0, multi_class='auto', solver='liblinear', max_iter=10000).fit(zs, ys)
    print('extract mean feature of testing data..')
    zs_test, ys_test = evaluator.extract_features(train=False)
    print('testing lr classifier..')
    accu = lr.score(zs_test, ys_test)
    print('mean accuray=%.4f' % accu)
    return accu

def train_knn_classifier(evaluator):
    print('extract mean feature of training data..')
    zs, ys = evaluator.extract_features(train=True)
#     num_classes = len(np.unique(ys))
    print('start to train knn classifier..')
    knn = KNeighborsRegressor().fit(zs, ys)
    print('extract mean feature of testing data..')
    zs_test, ys_test = evaluator.extract_features(train=False)
    print('testing knn classifier..')
    accu = knn.score(zs_test, ys_test)        
    print('mean accuray=%.4f' % accu)
    
def train_mlp_clf(dataset, data_dir):
    print('extract training data..')
    x, y = load_data_as_array(dataset, data_dir, train=True)
    print('start to train lr classifier..')
    clf = MLPClassifier(hidden_layer_sizes=(200,)).fit(x, y)
    print('extract testing data..')
    x_test, y_test = load_data_mlpclf(dataset, data_dir, train=False)
    print('testing lr classifier..')
    accu = clf.score(x_test, y_test)
    print('mean accuray=%.4f' % accu)     

def draw_one_batch(num_shots, dataset, data_dir, train, normalize, flatten):
    """
    draw x examples per class, if the rand_inds is given, then select the examples
    according to rand_inds.
    """
    images_selected = []
    labels_selected = []
    inds = []
    images, labels = load_data_as_array(dataset, data_dir, train=train, normalize=normalize, flatten=flatten)
    classes = np.unique(labels)
    for i in range(len(classes)):
        images_class_k = images[(labels == classes[i])]
        ind_k = np.random.choice(len(images_class_k), num_shots)
        inds.append(ind_k)
        images_selected.append(images_class_k[ind_k])
        labels_selected.append(np.ones(num_shots) * classes[i])
    return torch.Tensor(np.concatenate(images_selected, 0)), torch.Tensor(np.concatenate(labels_selected, 0))
    
def compute_nns(zs_train_data, zs_image_samples, train_data, dataset):
    sq_n2 = ((zs_train_data - zs_image_samples) ** 2).sum(-1)
    indices = torch.argmin(sq_n2, dim=-1) # .view(num_walks, num_images)
    nearestneighbours = torch.Tensor(train_data)[indices]
    if dataset == 'mnist' or dataset == 'fashionmnist':
        nearestneighbours = nearestneighbours.unsqueeze(1)
#     elif dataset == 'svhn':
#         nearestneighbours = nearestneighbours
#     else:
#         nearestneighbours = nearestneighbours.permute(0, 3, 1, 2)  
        
    return nearestneighbours

def plot_evolving_samples(images, nearestneighbours, fs=10, save_name=None):    
    print('plotting evolution of samples..')
    num_cols = images.shape[0]
    num_rows = images.shape[1] * 2
    images = images.squeeze().cpu().detach()
    if images.min() < 0.0:
        images = torch.clamp(images, min=-1, max=1)
        images = images * 0.5 + 0.5
    
    nearestneighbours = nearestneighbours.squeeze().cpu().detach()
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=0.95, wspace=0, hspace=0)
    fig = plt.figure(figsize=(fs, fs * num_rows / num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            ax = fig.add_subplot(gs[i, j])
            if i % 2 == 0:
                try:
                    ax.imshow(images[j, int(i/2)], cmap='gray', vmin=0, vmax=1.0)
                except:
                    ax.imshow(np.transpose(images[j, int(i/2)], (1,2,0)), vmin=0, vmax=1.0)
        
            else:
                try:
                    ax.imshow(nearestneighbours[j, int((i-1)/2)], cmap='gray', vmin=0, vmax=1.0)
                except:
                    ax.imshow(np.transpose(nearestneighbours[j, int((i-1)/2)], (1,2,0)), vmin=0, vmax=1.0)   

            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.vlines(0, ymin=0, ymax=28, linewidth=2, colors=('#009988' if i % 2 ==0 else '#EE7733'))
#                 rect = patches.Rectangle((0,0), 27*10, 27, linewidth=2, edgecolor=('#009988' if i % 2 ==0 else '#EE7733'),facecolor='none')            
#                 ax.add_patch(rect)
    plt.suptitle('---> Random Walks', fontsize=10)
    if save_name is not None:
        plt.savefig('samples/' + save_name + '_walks.png', dpi=300)
        plt.close()
        
        
class Evaluator_CLF():
    """
    evaluator for a supervised classifier
    """
    def __init__(self, clf, device, dataset, data_dir):
        super(self.__class__, self).__init__()
        self.clf = clf
        self.device = device
        self.dataset = dataset
        self.data_dir = data_dir
        if dataset == 'mnist' or dataset =='fashionmnist':
            self.input_channels, self.im_height, self.im_width = 1, 28, 28
        else:
            self.input_channels, self.im_height, self.im_width = 3, 32, 32 
    
    def test_accuracy(self):
        print('Computing the test accuracy..')
        test_data, img_dims = load_data(self.dataset, self.data_dir, batch_size=1000, train=False, normalize=False)
        N = 0.0
        accuracy = 0.0
        for b, (images, labels) in enumerate(test_data):
            N += len(labels)
            images = images.cuda().to(self.device)
            labels = labels.cuda().to(self.device)
            pred_labels = self.clf.forward(images)
            assert pred_labels.shape == (len(labels), 10), 'pred_y shape = %s' % pred_y.shape
            score = self.clf.score(pred_labels, labels)    
            accuracy += score.item()
        accuracy /= N
        print('Dataset=%s, Test Accuracy=%.4f' % (self.dataset, accuracy))            
        
        
        
        
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
            
    def oodauc(self, dataset_ood, score):
        """
        score : energy or gradient
        """
        if score == 'energy':
            score_ind = - self.extract_energy(self.dataset, train=False)
            score_ood = - self.extract_energy(dataset_ood, train=False)
        elif score == 'gradient':
            score_ind = - self.extract_pm(self.dataset, train=False)
            score_ood = - self.extract_pm(dataset_ood, train=False)
        else:
            raise NotImplementError
        labels_ind = np.ones_like(score_ind)
        labels_ood = np.zeros_like(score_ood)
        scores = np.concatenate([score_ind, score_ood])
        labels = np.concatenate([labels_ind, labels_ood])
        return roc_auc_score(labels, scores)
                             
    def nn_latents(self, image_samples):
        """
        given a set of image samples, compute their nearest neighbours in the training set.
        """
        print('compute nearest neighbours..')
        train_data, _ = load_data_as_array(self.dataset, self.data_dir, train=True, normalize=False, flatten=False, shuffle=False)
        zs_train_data, _ = self.extract_features(train=True, shuffle=False)
        
        num_walks, num_images, num_channels, H, W = image_samples.shape
        neural_ss1, neural_ss2 = self.ebm.forward(image_samples.view(num_walks * num_images, num_channels, H, W))
        zs_samples, _ = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
        zs_train_data = torch.Tensor(zs_train_data).unsqueeze(0).repeat(num_walks*num_images, 1, 1)
        
        zs_samples = zs_samples.unsqueeze(1).repeat(1, zs_train_data.shape[1], 1).cpu().detach()
        assert zs_train_data.shape == zs_samples.shape, "unexpected shape."
        nns = compute_nns(zs_train_data, zs_samples, train_data, dataset=self.dataset)
        return nns.view(num_walks, num_images, nns.shape[1], nns.shape[-2], nns.shape[-1])
    
    def uncond_sampling(self, batch_size, sgld_steps, sgld_lr, sgld_noise_std, grad_clipping=False, init_samples=None, logging_interval=None):
        print('sample unconditionally from ebm..')
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
        
        images_ebm = sgld_sampler.sample(ebm=self.ebm, 
                                         batch_size=batch_size, 
                                         num_steps=sgld_steps,
                                         pcd=False,
                                         init_samples=init_samples,
                                         logging_interval=logging_interval)
        if init_samples is not None:
            images_ebm = torch.cat((init_samples.unsqueeze(0), images_ebm), 0)
        return images_ebm

    def plot_final_samples(self, images, fs=10, save=False):
        print('plotting the samples..')
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
        if save:
            plt.savefig('samples/final_samples_%s.png' % self.dataset, dpi=300)
            plt.close()

    def plot_all_samples(self, list_images, fs=10, save=False):
        print('plotting the samples..')
        for j in range(len(list_images)):
            images = list_images[j]
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
            if save:
                plt.savefig('samples/%02d_samples_%s.png' % (j+1, self.dataset), dpi=300)
                plt.close()
            
    def extract_energy(self, dataset, train):
        print('Loading dataset=%s...' % dataset)
        test_data, img_dims = load_data(dataset, self.data_dir, 1000, train=train, normalize=True)
        Es = []
        for (images, _) in test_data:
            images = images.cuda().to(self.device)
            images = images + self.data_noise_std * torch.randn_like(images)
            energy = self.ebm.energy(images)
            Es.append(energy.cpu().detach().numpy())
        Es =  np.concatenate(Es, 0)
        return Es

    def extract_pm(self, dataset, train):
        print('Loading dataset=%s...' % dataset)
        test_data, img_dims = load_data(dataset, self.data_dir, 1000, train=train, normalize=True)
        Gs = []
        for (images, _) in test_data:
            images = images.cuda().to(self.device)
            images = images + self.data_noise_std * torch.randn_like(images)
            images.requires_grad = True
            grads = torch.autograd.grad(outputs=self.ebm.energy(images).sum(), inputs=images)[0]
            gradient_mass = torch.norm(torch.flatten(grads, start_dim=1), dim=1).squeeze()
            assert gradient_mass.shape == (len(images), ), "unexpected shape."
            Gs.append(gradient_mass.cpu().detach().numpy())
        Gs =  np.concatenate(Gs, 0)
        return Gs
    
    def extract_features(self, train, sample_size=None, shuffle=True):
        print('Loading dataset=%s...' % self.dataset)
        test_data, img_dims = load_data(self.dataset, self.data_dir, 1000, train=train, normalize=True, shuffle=shuffle)
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

    def extract_features_fewshots(self, data):
        zs = []
        ys = []
        images = data['images']
        images = (images - 0.5) / 0.5
        labels = data['labels']
        images = images.cuda().to(self.device)
        images = images + self.data_noise_std * torch.randn_like(images)
        neural_ss1, neural_ss2 = self.ebm.forward(images)
        mean, sigma = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
        zs.append(mean.squeeze().cpu().detach().numpy())
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

    def plot_tsne(self, zs2, ys, fs=8, ms=10, save=False):
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
        if save:
            plt.savefig('tsne_%s.png' % self.dataset, dpi=300)           
        
    def plot_oods(self, dataset, train, fs=8, density=False, save=False):
        print('extract mean feature of data using EBM..')
        energy_id = self.extract_energy(dataset=self.dataset, train=train)
        energy_ood = self.extract_energy(dataset=dataset, train=train)
#         prior_mu, prior_sigma = nats_to_params(self.ebm.prior_nat1, self.ebm.prior_nat2)
#         prior_dist = Normal(prior_mu, prior_sigma)
#         zs = torch.Tensor(zs).cuda().to(self.device)
#         post_log_prob = prior_dist.log_prob(zs).sum(-1).cpu().detach().numpy()
#         prior_log_prob = prior_dist.log_prob(prior_dist.sample((len(ys), ))).sum(-1).cpu().detach().numpy()
        fig = plt.figure(figsize=(fs*2,fs))
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.hist(-energy_id, bins=100, label='in-dist (%s)' % self.dataset, alpha=.3, density=density)
        ax.set_title('Train on %s' % self.dataset, fontsize=14)
        _ = ax.hist(-energy_ood, bins=100, label='out-of-dist (%s)' % dataset , alpha=.3, density=density)
        ax.legend(fontsize=14)
        ax.set_xlabel('- E(x)')
        if save:
            plt.savefig('CEBM_OOD_in=%s_out=%s.png' % (self.dataset, dataset), dpi=300)    
            

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
                
    def plot_oods(self, dataset, train, score, sample_size=1000, batch_size=100, fs=8, density=False, save=False):
        print('OOD-Detection for VAE with %s..' % score)
        print('Computing score for ind data..')
        score_ind = self.extract_metrics(self.dataset, train, sample_size, batch_size, metric=score)   
        print('Computing score for ood data..')
        score_ood = self.extract_metrics(dataset, train, sample_size, batch_size, metric=score)        
        fig = plt.figure(figsize=(fs*2,fs))
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.hist(score_ind, bins=100, label='in-dist (%s)' % self.dataset, alpha=.3, density=density)
        ax.set_title('Train on %s' % self.dataset, fontsize=14)
        _ = ax.hist(score_ood, bins=100, label='out-of-dist (%s)' % dataset , alpha=.3, density=density)
        ax.legend(fontsize=14)
        ax.set_xlabel('- E(x)')
        if save:
            plt.savefig('VAE_OOD_in=%s_out=%s.png' % (self.dataset, dataset), dpi=300)    
          
        
    def test_one_batch(self, batch_size):
        test_data, _ = load_data(self.dataset, self.data_dir, batch_size=batch_size, train=False, normalize=False)
        for (images, _) in test_data:
            break
        images = images.cuda().to(self.device)
        latents, q_log_prob = self.enc.forward(images)
        recons, ll, p_log_prob = self.dec.forward(latents, images)
        recons = recons.view(batch_size, self.input_channels, self.im_height, self.im_width).squeeze()
        return images, recons
        
    def plot_samples(self, images, recons, fs=10, data_name=None):
        test_batch_size = len(images)
        images = images.squeeze().cpu().detach().numpy()
        recons = recons.squeeze().cpu().detach().numpy()
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
            
    def extract_features(self, train, shuffle=False):
        zs = []
        ys = []
        train_data, img_dims = load_data(self.dataset, self.data_dir, 1000, train=train, normalize=False, shuffle=shuffle)   
        for b, (images, labels) in enumerate(train_data):
            images = images.cuda().to(self.device)
            mu, _ = self.enc.enc_net(images)
            zs.append(mu.cpu().detach().numpy())
            ys.append(labels)
        zs = np.concatenate(zs, 0)
        ys = np.concatenate(ys, 0)
        return zs, ys   
    
    def extract_metrics(self, dataset, train, sample_size, batch_size, metric, shuffle=False):
        Metric = []
        train_data, img_dims = load_data(dataset, self.data_dir, batch_size, train=train, normalize=False, shuffle=shuffle)   
        for b, (images, labels) in enumerate(train_data):
            _, input_c, im_h, im_w = images.shape
            images = images.repeat(sample_size, 1, 1, 1, 1).view(sample_size*batch_size, input_c, im_h, im_w).cuda().to(self.device)            
            latents, q_log_prob = self.enc.forward(images)
            recon, ll, p_log_prob = self.dec.forward(latents, images)
            log_w = (ll + p_log_prob - q_log_prob)
            log_w = log_w.view(sample_size, batch_size)
            if metric == 'elbo':
                Metric.append(log_w.mean(0).detach().cpu().numpy())
            elif metric == 'marginal':
                Metric.append(torch.logsumexp(log_w, 0).detach().cpu().numpy() - np.log(np.array([sample_size])))
        Metric = np.concatenate(Metric, 0)
        return Metric
    
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

    def random_walks(self, num_random_walks, images, train=False, sample=False):
        list_images = []
        list_images.append(images.unsqueeze(0))
        images = images.cuda().to(self.device)
        for i in range(num_random_walks):
            if sample:
                latents, _ = self.enc.forward(images)
                recons, _, _  = self.dec.forward(latents, images)
            else:
                mu, _ = self.enc.enc_net(images)
                recons, _, _  = self.dec.forward(mu, images)
            list_images.append(recons.cpu().detach().unsqueeze(0))
        return torch.cat(list_images, 0)
    
    def nn_latents(self, image_samples):
        """
        given a set of image samples, compute their nearest neighbours in the training set.
        """
        print('compute nearest neighbours..')
        train_data, _ = load_data_as_array(self.dataset, self.data_dir, train=True, normalize=False, flatten=False, shuffle=False)
        zs_train_data, _ = self.extract_features(train=True, shuffle=False)
        num_walks, num_images, num_channels, H, W = image_samples.shape
        zs_samples, _ = self.enc.enc_net(image_samples.view(num_walks * num_images, num_channels, H, W).cuda().to(self.device))
        zs_train_data = torch.Tensor(zs_train_data).unsqueeze(0).repeat(num_walks*num_images, 1, 1)
        zs_samples = zs_samples.unsqueeze(1).repeat(1, zs_train_data.shape[1], 1).cpu().detach()
        assert zs_train_data.shape == zs_samples.shape, "unexpected shape."
        nns = compute_nns(zs_train_data, zs_samples, train_data, dataset=self.dataset)
        return nns.view(num_walks, num_images, nns.shape[1], nns.shape[-2], nns.shape[-1])