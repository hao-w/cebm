import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from cebm.data import setup_data_loader
from cebm.utils import load_models
from cebm.sgld import SGLD_Sampler

def plot_samples(images, denormalize, fs=1, save=False):
    test_batch_size = len(images)
    images = images.squeeze().cpu().detach()
    images = torch.clamp(images, min=-1, max=1)
    if denormalize:
        images = images * 0.5 + 0.5
    gs = gridspec.GridSpec(int(test_batch_size/10), 10)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.1, hspace=0.1)
    fig = plt.figure(figsize=(fs*10, fs*int(test_batch_size/10)))
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
    
def generate_samples(self, batch_size, **kwargs):
    if self.model_name in ['IGEBM', 'CEBM', 'CEBM_GMM']:
        raise NotImplementedError()
  
    elif self.model_name in ['VAE', 'VAE_GMM', 'BIGAN', 'BIGAN_GMM']:
        return self.models['enc'].latent_params()
    else:
        raise NotImplementError
            
def uncond_sampling(models, sgld_steps, batch_size, sgld_args, init_samples=None, save=False):
    print('sample unconditionally from ebm..')
    sgld_sampler = SGLD_Sampler(**sgld_args)
    images_ebm = sgld_sampler.sample(ebm=models['ebm'], 
                                     batch_size=batch_size, 
                                     num_steps=sgld_steps,
                                     pcd=False,
                                     init_samples=init_samples)
    plot_samples(images_ebm.tanh(), denormalize=True, save=save)    
    
class Evaluator():
    def __init__(self, device, models, model_name, data, data_dir='../datasets/', **kwargs):
        super().__init__()
        self.models = models
        self.model_name = model_name
        self.data = data
        self.data_dir = data_dir
        self.device = device
        if self.model_name in ['IGEBM', 'CEBM', 'CEBM_GMM']:
            self.sgld_sampler = kwargs['sgld_sampler']
        
    def latent_mode(self, images):
        if self.model_name in ['IGEBM', 'CEBM', 'CEBM_GMM', 'IGEBM_VERA', 'CEBM_VERA', 'CEBM_GMM_VERA']:
            return self.models['ebm'].latent_params(images)
        elif self.model_name in ['VAE', 'VAE_GMM', 'BIGAN', 'BIGAN_GMM']:
            return self.models['enc'].latent_params(images)
        else:
            raise NotImplementError
            
    def encode_dataset(self, data_loader):
        """
        return the modes of latent representations with the correspoding labels
        """
        zs = []
        ys = []
        for b, (images, labels) in enumerate(data_loader):
            images = images.to(self.device)
            if self.model_name in ['VAE', 'VAE_GMM']:
                images = images.unsqueeze(0)
            mean, _ = self.latent_mode(images)
            zs.append(mean.squeeze(0).detach().cpu().numpy())
            ys.append(labels.numpy())
        zs = np.concatenate(zs, 0)
        ys = np.concatenate(ys, 0)
        return zs, ys
            
    
    def few_label_classification(self, list_num_shots=[1, 10, 100, -1], num_runs=10, batch_size=1000, classifier='logistic'):
        """
        Import classifier implementations from scikit-learn  
        train logistic classifiers with the encoded representations of the training set
        compute the accuracy for the test set
        """
        if not os.path.exists('results/few_label/'):
            os.makedirs('results/few_label/')
        results = {'Num_Shots': [], 'Mean': [], 'Std': []}
        for num_shots in tqdm(list_num_shots):
            Accuracy = []
            for i in range(num_runs):
                torch.cuda.empty_cache()
                if num_shots == -1:
                    train_loader, im_h, im_w, im_channels = setup_data_loader(data=self.data, 
                                                                              data_dir=self.data_dir, 
                                                                              num_shots=num_shots, 
                                                                              batch_size=batch_size, 
                                                                              train=True, 
                                                                              normalize=False if self.model_name in ['VAE', 'VAE_GMM'] else True, 
                                                                              shuffle=False, 
                                                                              shot_random_seed=None if num_shots==-1 else i)
                    zs_train, ys_train = self.encode_dataset(train_loader)
                else:
                    data = torch.load('/home/hao/Research/cebm/cebm/datasets/fewshots/%s/%d/%d.pt' % (self.data, num_shots*10, i+1))
                    images = ((data['images'] - 0.5) / 0.5).to(self.device)
                    zs_train, _ = self.models['ebm'].latent_params(images)
                    zs_train = zs_train.cpu().detach().numpy()
                    ys_train = data['labels'].numpy()
                if classifier == 'logistic':
                    clf = LogisticRegression(random_state=0, 
                                             multi_class='auto', 
                                             solver='liblinear', 
                                             max_iter=10000).fit(zs_train, ys_train)
                    
                else:
                    raise NotImplementedError
                torch.cuda.empty_cache()
                test_loader, im_h, im_w, im_channels = setup_data_loader(data=self.data, 
                                                                         data_dir=self.data_dir, 
                                                                         num_shots=num_shots, 
                                                                         batch_size=batch_size, 
                                                                         train=False, 
                                                                         normalize=False if self.model_name in ['VAE', 'VAE_GMM'] else True, 
                                                                         shuffle=False, 
                                                                         shot_random_seed=None if num_shots==-1 else i)
                zs_test, ys_test = self.encode_dataset(test_loader)
                Accuracy.append(np.array([clf.score(zs_test, ys_test)]))
                #Only run it once when using the fulling training set
                if num_shots == -1:
                    break
            Accuracy = np.concatenate(Accuracy)
            results['Num_Shots'].append(num_shots)
            results['Mean'].append(Accuracy.mean())
            results['Std'].append(Accuracy.std())
        pd.DataFrame.from_dict(results).to_csv('results/few_label/%s-%s-%s.csv' % (self.data, num_shots*10, i+1), index=False)
        return results
#         print('clf=%s, model=%s, data=%s, num_shots=%d, mean=%.2f, std=%.2f' % (classifier, self.model_name, self.data)
#         fout.close()
    
#     def train_logistic_classifier(zs, ys, ):
#         accu = lr.score(zs_test, ys_test)
#     #     print('mean accuray=%.4f' % accu)
#         return accu
 
        