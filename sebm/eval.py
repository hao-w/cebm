import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sebm.data import load_data, load_data_as_array
from sebm.sgld import SGLD_sampler
from sebm.gaussian_params import nats_to_params, params_to_nats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import time
from tqdm import tqdm
from sebm.supervised_clf import Downstream_Semi_Clf
from sebm.models import Clf

def plot_likelihood_cond_samples(images_ebm, fs=2, save_name=None):    
    num_rows, num_cols = len(images_ebm[0]), len(images_ebm)
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=0.95, wspace=0.3, hspace=0.3)
    fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            ax = fig.add_subplot(gs[i, j])
            images_b = images_ebm[j].cpu().detach()
            if images_b.min() < 0.0:
                images_b = torch.clamp(images_b, min=-1, max=1)
                images_b = images_b * 0.5 + 0.5
                images_b = images_b.numpy()
            try:
                ax.imshow(images_b[i], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(images_b[i], (1,2,0)), vmin=0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
    if save_name is not None:
        plt.savefig('samples/' + save_name + '_ll_samples.png', dpi=300)
        plt.close()
        
def plot_nearest_neighbors(test_data, nns, min_labels, min_distances, fs=2, save_name=None):    
    num_rows, num_cols = nns.shape[0], nns.shape[1]+1
    true_images = test_data['images']
    true_labels = test_data['labels']
    if true_images.min() < 0.0:
        true_images = torch.clamp(true_images, min=-1, max=1)
        true_images = true_images * 0.5 + 0.5
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.3, hspace=0.3)
    fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
    
    true_images = true_images.squeeze(1).numpy()
    nns = nns.squeeze(2).numpy()
    for i in range(num_rows):
        ax = fig.add_subplot(gs[i, 0])
        try:
            ax.imshow(true_images[i], cmap='gray', vmin=0, vmax=1.0)
        except:
            ax.imshow(np.transpose(true_images[i], (1,2,0)), vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Class=%d' % true_labels[i], fontsize=8)
        for j in range(1, num_cols):
            ax = fig.add_subplot(gs[i, j])
            try:
                ax.imshow(nns[i, j-1], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(nns[i, j-1], (1,2,0)), vmin=0, vmax=1.0)   
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Class=%d' % (min_labels[i, j-1]), fontsize=8)
    if save_name is not None:
        plt.savefig('samples/' + save_name + '_nns.png', dpi=300)
        plt.close()

        
def plot_k_nearest_neighbors(test_images, test_labels, nns, min_labels, min_distances, fs=2, save_name=None):    
    num_rows, num_cols = nns.shape[0], 4
#     true_images = test_data['images']
#     true_labels = test_data['labels']
#     if true_images.min() < 0.0:
#         true_images = torch.clamp(true_images, min=-1, max=1)
#         true_images = true_images * 0.5 + 0.5
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
    
    test_images = test_images.squeeze(1).numpy()
    nns = nns.squeeze(2).numpy()
    for i in range(num_rows):
        ax = fig.add_subplot(gs[i, 0])
        try:
            ax.imshow(test_images[i], cmap='gray', vmin=0, vmax=1.0)
        except:
            ax.imshow(np.transpose(test_images[i], (1,2,0)), vmin=0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
#         ax.set_title('Class=%d' % test_labels[i], fontsize=8)
        for j in range(1, num_cols):
            ax = fig.add_subplot(gs[i, j])
            try:
                ax.imshow(nns[i, j-1], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(nns[i, j-1], (1,2,0)), vmin=0, vmax=1.0)   
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
#             ax.set_title('Class=%d' % (min_labels[i, j-1]), fontsize=8)
    if save_name is not None:
        plt.savefig('samples/' + save_name + '_nns.svg')
        plt.close()

        
def semi_nn_clf(model_name, device, evaluator, list_num_shots=[1, 10, 100, -1], num_runs=10, num_epochs=100, batch_size=100, reg_lambda=1e-5):
    fout = open('semi-nn-clf-%s.txt' % model_name, 'a+')
    for num_shots in tqdm(list_num_shots):
        Accu = []
        for i in range(num_runs):
            zs, ys = unlabel_latents(evaluator, num_shots, seed=i)
            zs, ys = torch.Tensor(zs), torch.LongTensor(ys)
            classifier = Downstream_Semi_Clf(device, reg_lambda, batch_size=batch_size, num_epochs=num_epochs)
            classifier.train(zs, ys)
            test_zs, test_ys = unlabel_latents(evaluator, num_shots=-1, seed=i, train=False)
            test_zs, test_ys = torch.Tensor(test_zs), torch.LongTensor(test_ys)
            accu = classifier.score(test_zs, test_ys).item()
            print(accu)
            Accu.append(np.array([accu]))
            if num_shots == -1:
                break
        Accu = np.concatenate(Accu)
        print('data=%s, num_shots=%d, mean=%.4f, std=%.4f' % (evaluator.dataset, num_shots, Accu.mean(), Accu.std()), file=fout)
    fout.close()
    
def fewshots(model_name, evaluator, list_num_shots=[1, 10, 100, -1], num_runs=10):
    fout = open('fewshots-%s.txt' % model_name, 'a+')
    for num_shots in tqdm(list_num_shots):
        Accu = []
        for i in range(num_runs):
            if num_shots == -1:
                accu = train_logistic_classifier(evaluator, train_data=None)
                Accu.append(np.array([accu]))
                break
            else:
                data = torch.load('/home/hao/Research/sebm_data/fewshots/%s/%d/%d.pt' % (evaluator.dataset, num_shots*10, i+1))
                accu = train_logistic_classifier(evaluator, train_data=data)
                Accu.append(np.array([accu]))
            
        Accu = np.concatenate(Accu)
        print('data=%s, num_shots=%d, mean=%.4f, std=%.4f' % (evaluator.dataset, num_shots, Accu.mean(), Accu.std()), file=fout)
    fout.close()
        
def unlabel_latents(evaluator, num_shots, seed, train=True):
    ys_permuted = []
    zs_permuted = []
    np.random.seed(seed)
    zs, ys = evaluator.extract_features(train=train, shuffle=False)
    if num_shots == -1:
        return zs, ys
    else:
        classes = np.unique(ys)
        for k in range(len(classes)):
            ys_k = ys[(ys == classes[k])]
            zs_k = zs[(ys == classes[k])]
            ind_k = np.random.permutation(np.arange(len(ys_k)))
            ys_k = ys_k[ind_k]
            zs_k = zs_k[ind_k]
            ys_k[num_shots:] = -1
            ys_permuted.append(ys_k)
            zs_permuted.append(zs_k)
        return np.concatenate(zs_permuted, 0), np.concatenate(ys_permuted, 0)
    
def label_propagation(algo_name, evaluator, num_shots, seed, kernel, gamma, n_neighbors, max_iter):
    ts = time.time()
    zs_train, ys_train = unlabel_latents(evaluator, num_shots, seed)
    if algo_name == 'lp':
#         print('training label propagation model..')
        lp = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter).fit(zs_train, ys_train)
    elif algo_name == 'ls':
#         print('training label spreading model..')
        lp = LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter).fit(zs_train, ys_train)
    else:
        raise NotImplementError
#     print('training completed in %ds. testing..' % (te - ts))
    zs_test, ys_test = evaluator.extract_features(train=False, shuffle=False)
    accu = lp.score(zs_test, ys_test)
    te = time.time()
    print('(%ds) mean accuracy=%s' % (te-ts, accu))
    return accu

def similarity_z_space_fewshots(evaluator, train_batch_size, test_data):
    """
    For each test data point, compute the L2 norm distance from the training data in latent space,
    return the confusion matrix
    """
    zs_train, ys_train = evaluator.extract_features(train=True, shuffle=False)
    zs_test, ys_test = evaluator.extract_features_fewshots(data=test_data)
    zs_train = torch.Tensor(zs_train)
    ys_train = torch.Tensor(ys_train)
    zs_test = torch.Tensor(zs_test)
    ys_test = torch.Tensor(ys_test)
    test_batch_size = len(ys_test)
    zs_test = zs_test.unsqueeze(1).repeat(1, train_batch_size, 1)
    distances = []
    for i in range(int(len(ys_train) / train_batch_size)):
        zs_train_b = zs_train[(i)*train_batch_size:(i+1)*train_batch_size]                
        zs_train_b = zs_train_b.repeat(test_batch_size, 1, 1)
        sq_norm = ((zs_train_b - zs_test) ** 2).sum(-1)
        distances.append(sq_norm)
    distances = torch.cat(distances, 1)
    minimum_distances, indicies = torch.topk(distances, k=3, dim=-1, largest=False, sorted=True)
    min_labels = ys_train[indicies]
    train_data, _ = load_data_as_array(evaluator.dataset, evaluator.data_dir, train=True, normalize=False, flatten=False, shuffle=False)
    train_data = torch.Tensor(train_data).repeat(test_batch_size, 1, 1, 1, 1) 
    indicies_expand = indicies.unsqueeze(-1).repeat(1, 1, train_data.shape[2]).unsqueeze(-1).repeat(1, 1, 1, train_data.shape[3]).unsqueeze(-1).repeat(1,1,1,1,train_data.shape[4])
    nns = torch.gather(train_data, 1, indicies_expand)
    return minimum_distances, min_labels, nns

def similarity_z_space(evaluator, train_batch_size, test_batch_size, model_name):
    """
    For each test data point, compute the L2 norm distance from the training data in latent space,
    return the confusion matrix
    """
    zs_train, ys_train = evaluator.extract_features(train=True, shuffle=False)
    zs_test, ys_test = evaluator.extract_features(train=False, shuffle=False)
    zs_train = torch.Tensor(zs_train)
    zs_test = torch.Tensor(zs_test)
    ys_train = torch.Tensor(ys_train)
    ys_test = torch.Tensor(ys_test)
    pred_ys_test = []
    for b in range(int(len(ys_test) / test_batch_size)):
        zs_test_b = zs_test[(b)*test_batch_size:(b+1)*test_batch_size]
        zs_test_b = zs_test_b.unsqueeze(1).repeat(1, train_batch_size, 1)
        distances = []
        for i in range(int(len(ys_train) / train_batch_size)):
            zs_train_b = zs_train[(i)*train_batch_size:(i+1)*train_batch_size]                
            zs_train_b = zs_train_b.repeat(test_batch_size, 1, 1)
            sq_norm = ((zs_train_b - zs_test_b) ** 2).sum(-1)
            distances.append(sq_norm)
        distances = torch.cat(distances, 1)
        indices = torch.argmin(distances, dim=-1) # test_size
        pred_ys_test.append(ys_train[indices].unsqueeze(-1))
        print('%d completed..' % (b+1))
    pred_ys_test = torch.cat(pred_ys_test, 0)
    paired = torch.cat((ys_test.unsqueeze(-1), pred_ys_test), -1)
    torch.save(paired, 'confusion_matrix_labels_%s_z_%s.pt' % (model_name, evaluator.dataset))

def similarity_pixel_space_fewshots(train_dataset, test_data, data_dir, train_batch_size):
    """ 
    """
    train_data, _ = load_data(train_dataset, data_dir, batch_size=train_batch_size, train=True, shuffle=False, normalize=False)
    test_images = test_data['images']
    test_images = torch.flatten(test_images, start_dim=1)
    test_images = test_images.unsqueeze(1).repeat(1, train_batch_size, 1)
    distances = []
    train_labels = torch.Tensor(train_data.dataset.targets)
    train_images = torch.Tensor(np.transpose(train_data.dataset.data, (0, 3, 1, 2)).astype(float))
    test_batch_size = len(test_images)
    for (train_images_b, _) in train_data:
        train_images_b = torch.flatten(train_images_b, start_dim=1).repeat(test_batch_size, 1, 1)
        sq_norm = ((train_images_b - test_images) ** 2).sum(-1)
        distances.append(sq_norm)
    distances = torch.cat(distances, 1)
    minimum_distances, indicies = torch.topk(distances, k=3, dim=-1, largest=False, sorted=True)
    min_labels = train_labels[indicies]
#     train_data, _ = load_data_as_array(dataset, data_dir, train=True, normalize=False, flatten=False, shuffle=False)
    train_images = train_images.repeat(test_batch_size, 1, 1, 1, 1) 
    train_images /= 255.0
    indicies_expand = indicies.unsqueeze(-1).repeat(1, 1, train_images.shape[2]).unsqueeze(-1).repeat(1, 1, 1, train_images.shape[3]).unsqueeze(-1).repeat(1,1,1,1,train_images.shape[4])
    nns = torch.gather(train_images, 1, indicies_expand)
#     nns = train_images[torch.arange(test_batch_size), indices]
    return minimum_distances, min_labels, nns 

def similarity_pixel_space(train_dataset, test_dataset, data_dir, train_batch_size, test_batch_size):
    """
    For each test data point, compute the L2 norm distance from the training data in the pixel space,
    return the confusion matrix
    """
    train_data, _ = load_data(train_dataset, data_dir, batch_size=train_batch_size, train=True, shuffle=False, normalize=False)
    test_data, _ = load_data(test_dataset, data_dir, batch_size=test_batch_size, train=False, shuffle=False, normalize=False)
    all_test_labels = []
    pred_test_labels = []
    for b, (test_images, test_labels) in enumerate(test_data):
        test_images = torch.flatten(test_images, start_dim=1)
        test_images = test_images.unsqueeze(1).repeat(1, train_batch_size, 1)
        all_test_labels.append(test_labels.unsqueeze(-1))
        distances = []
        all_train_labels = []
        for (train_images, train_labels) in train_data:
            train_images = torch.flatten(train_images, start_dim=1)
            train_images = train_images.repeat(test_batch_size, 1, 1)
            sq_norm = ((train_images - test_images) ** 2).sum(-1)
            distances.append(sq_norm)
            all_train_labels.append(train_labels.unsqueeze(-1))
        distances = torch.cat(distances, 1)
        indices = torch.argmin(distances, dim=-1) # test_size
        all_train_labels = torch.cat(all_train_labels, 0).squeeze(-1) # train_size
        pred_test_labels.append(all_train_labels[indices].unsqueeze(-1))
        print('%d completed..' % (b+1))
    all_test_labels = torch.cat(all_test_labels, 0)
    pred_test_labels = torch.cat(pred_test_labels, 0)
    paired = torch.cat((all_test_labels, pred_test_labels), -1)
    torch.save(paired, 'confusion_matrix_labels_pixel_%s.pt' % train_dataset)

def train_logistic_classifier(evaluator, train_data=None):
    if train_data is None:
#         print('extract mean feature of training data..')
        zs, ys = evaluator.extract_features(train=True)
    else:
        zs, ys = evaluator.extract_features_fewshots(train_data)
#     print('start to train lr classifier..')
    lr = LogisticRegression(random_state=0, multi_class='auto', solver='liblinear', max_iter=10000).fit(zs, ys)
#     print('extract mean feature of testing data..')
    zs_test, ys_test = evaluator.extract_features(train=False)
#     print('testing lr classifier..')
    accu = lr.score(zs_test, ys_test)
#     print('mean accuray=%.4f' % accu)
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
    
    def similarity_ebm_density_space(self, train_batch_size, test_batch_size):
        """
        For each test data point, compute the L2 norm distance from the training data in latent space,
        return the confusion matrix
        """
        zs_test, ys_test = self.extract_features(train=False, shuffle=False)
        zs_test = torch.Tensor(zs_test)
        ys_test = torch.Tensor(ys_test)
        pred_ys_test = []
        train_data, _ = load_data(self.dataset, self.data_dir, train_batch_size, train=True, normalize=True, shuffle=False)
        ys_train = torch.Tensor(train_data.dataset.targets)
        for b in range(int(len(ys_test) / test_batch_size)):
            zs_test_b = zs_test[(b)*test_batch_size:(b+1)*test_batch_size]
            zs_test_b = zs_test_b.unsqueeze(1).repeat(1, train_batch_size, 1)
            densities = []
            zs_test_b = zs_test_b.cuda().to(self.device)
            for i, (train_images, train_labels) in enumerate(train_data):
                train_images = train_images.cuda().to(self.device)
                log_prior = self.ebm.log_prior(zs_test_b) # test_size * train_size
                ll = self.ebm.log_factor_expand(train_images, zs_test_b, test_batch_size)
                log_joint = (ll + log_prior).cpu().detach()
                densities.append(log_joint)
            indices = torch.argmax(torch.cat(densities, 1), dim=-1) # test_size
            pred_ys_test.append(ys_train[indices].unsqueeze(-1))
            print('%d completed..' % (b+1))
        pred_ys_test = torch.cat(pred_ys_test, 0)
        return ys_test, pred_ys_test
    
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

    
    def uncond_sampling_ll(self, sample_size, batch_size, sgld_steps, sgld_lr, sgld_noise_std, init_images, grad_clipping=False, logging_interval=None):
        print('encoding initial images..')
        init_images = init_images.cuda().to(self.device)
        init_images = init_images + self.data_noise_std * torch.randn_like(init_images)
        neural_ss1, neural_ss2 = self.ebm.forward(init_images)
        mean_z, _ = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
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
        list_images = []
        list_images.append(init_images)
        for i in range(sample_size):
            images_ebm = sgld_sampler.sample_ll(ebm=self.ebm, 
                                             latents=mean_z,
                                             batch_size=batch_size, 
                                             num_steps=sgld_steps,
                                             init_samples=None,
                                             logging_interval=logging_interval)
            list_images.append(images_ebm)
#             images_ebm = images_ebm + self.data_noise_std * torch.randn_like(init_images)
#             neural_ss1, neural_ss2 = self.ebm.forward(images_ebm)
#             mean_z, _ = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
        return list_images
    
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
        num_cols = len(list_images)
        num_rows = list_images[0].shape[0]
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
        fig = plt.figure(figsize=(fs*num_cols, fs*num_rows))
        
        for j, images in enumerate(list_images):
            images = images.squeeze().cpu().detach()
            images = torch.clamp(images, min=-1, max=1)
            images = images * 0.5 + 0.5
            for i in range(len(images)):
#                 print(i)
                ax = fig.add_subplot(gs[i, j])
                try:
                    ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
                except:
                    ax.imshow(np.transpose(images[i].numpy(), (1,2,0)), vmin=0, vmax=1.0)
                ax.set_axis_off()
                ax.set_xticks([])
                ax.set_yticks([])
            if save:
                plt.savefig('samples/%s_ll_sampling.png' % (self.dataset), dpi=300)
                plt.close()
            
    def extract_energy(self, dataset, train):
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
    
    def extract_features(self, train, shuffle=False):
        test_data, img_dims = load_data(self.dataset, self.data_dir, 1000, train=train, normalize=True, shuffle=shuffle)
        zs = []
        ys = []
        for (images, labels) in test_data:
            images = images.cuda().to(self.device)
#             images = images + self.data_noise_std * torch.randn_like(images)
            try:
                neural_ss1, neural_ss2 = self.ebm.forward(images)
                mean, sigma = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
            except:
                mean = self.ebm.forward(images)
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
#         images = images + self.data_noise_std * torch.randn_like(images)
        try:
            neural_ss1, neural_ss2 = self.ebm.forward(images)
            mean, sigma = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
        except:
            mean = self.ebm.forward(images)
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
    def __init__(self, enc, dec, arch, device, dataset, data_dir):
            super(self.__class__, self).__init__()
            self.enc = enc
            self.dec = dec
            self.arch = arch
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
        _ = ax.hist(score_ind, bins=100, label='in-dist (%s)' % self.dataset, density=density)
        ax.set_title('Train on %s' % self.dataset, fontsize=14)
        _ = ax.hist(score_ood, bins=100, label='out-of-dist (%s)' % dataset, density=density)
        ax.legend(fontsize=14)
        if score == 'elbo':
            ax.set_xlabel('elbo', fontsize=14)
        elif score == 'marginal':
            ax.set_xlabel('log_p(x)', fontsize=14)
        if save:
            plt.savefig('VAE_OOD_score=%s_in=%s_out=%s.png' % (score, self.dataset, dataset), dpi=300)    
          
        
    def test_one_batch(self, batch_size):
        """
        with only one sample
        """
        test_data, _ = load_data(self.dataset, self.data_dir, batch_size=batch_size, train=False, normalize=False)
        for (images, _) in test_data:
            break
        if self.arch == 'mlp':
            images = images.view(batch_size, input_c*im_h*im_w)
        images = images.unsqueeze(0).cuda().to(self.device)
        latents, q_log_prob = self.enc.forward(images)
        recons, ll, p_log_prob = self.dec.forward(latents, images)
        return images, recons
        
    def plot_samples(self, images, recons, fs=10, data_name=None):
        images = images.squeeze().cpu().detach().numpy()
        recons = recons.squeeze().cpu().detach().numpy()
        test_batch_size = len(images)
        gs = gridspec.GridSpec(int(test_batch_size/10)*2, 10)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs*10, fs))
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
    
    def extract_features_fewshots(self, data):
        zs = []
        ys = []
        images = data['images']
        labels = data['labels']
        images = images.cuda().to(self.device)
        mu, _ = self.enc.enc_net(images)
        zs.append(mu.cpu().detach().numpy())
        ys.append(labels)
        zs = np.concatenate(zs, 0)
        ys = np.concatenate(ys, 0)
        return zs, ys   
    
    def oodauc(self, dataset_ood, score, sample_size=1000, batch_size=100):
        """
        score : energy or gradient
        """
        if score == 'marginal':
            score_ind = - self.extract_metrics(self.dataset, False, sample_size, batch_size, metric='marginal')
            score_ood = - self.extract_metrics(dataset_ood, False, sample_size, batch_size, metric='marginal')
        elif score == 'gradient':
            score_ind = - self.extract_pm(self.dataset, False, sample_size, batch_size)
            score_ood = - self.extract_pm(dataset_ood, False, sample_size, batch_size)
        else:
            raise NotImplementError
        labels_ind = np.ones_like(score_ind)
        labels_ood = np.zeros_like(score_ood)
        scores = np.concatenate([score_ind, score_ood])
        labels = np.concatenate([labels_ind, labels_ood])
        return roc_auc_score(labels, scores)
    
    def extract_pm(self, dataset, train, sample_size, batch_size, shuffle=False):
        test_data, img_dims = load_data(dataset, self.data_dir, batch_size, train=train, normalize=False)
        Gs = []
        for (images, _) in test_data:
            images = images.cuda().to(self.device)
            images.requires_grad = True
            images = images.repeat(sample_size, 1, 1, 1, 1)
            latents, q_log_prob = self.enc.forward(images)
            _, ll, p_log_prob = self.dec.forward(latents, images)
            log_w = (ll + p_log_prob - q_log_prob)
#             log_w = log_w.view(sample_size, batch_size)
            pm = torch.logsumexp(log_w, 0).sum()
            grads = torch.autograd.grad(outputs=pm, inputs=images)[0]
#             print(grads.shape)
            gradient_mass = torch.norm(torch.flatten(grads, start_dim=2), dim=2).mean(0)
#             assert gradient_mass.shape == (batch_size, ), "unexpected shape."
            Gs.append(gradient_mass.cpu().detach().numpy())
        Gs =  np.concatenate(Gs, 0)
        return Gs
    
    def extract_metrics(self, dataset, train, sample_size, batch_size, metric, shuffle=False):
        Metric = []
        train_data, img_dims = load_data(dataset, self.data_dir, batch_size, train=train, normalize=False, shuffle=shuffle)   
        for b, (images, labels) in enumerate(train_data):
            images = images.repeat(sample_size, 1, 1, 1, 1).cuda().to(self.device)            
            latents, q_log_prob = self.enc.forward(images)
            _, ll, p_log_prob = self.dec.forward(latents, images)
            log_w = (ll + p_log_prob - q_log_prob)
#             log_w = log_w.view(sample_size, batch_size)
            if metric == 'elbo':
                Metric.append(log_w.mean(0).detach().cpu().numpy())
            elif metric == 'marginal':
                Metric.append(torch.logsumexp(log_w, 0).detach().cpu().numpy() - np.log(np.array([sample_size])))
            elif metric == 'bpd':
                log_e_px = torch.logsumexp(log_w, 0).detach().cpu().numpy() - np.log(np.array([sample_size]))
                bpd = - log_e_px / np.log(np.array([2])) / np.prod(np.array(img_dims))
                sigmax = torch.sigmoid(images[0].cpu()).numpy()
                extra = ((np.log(sigmax) / np.log(np.array([2]))) + (np.log(1 - sigmax) /  np.log(np.array([2])))).sum(-1).sum(-1).sum(-1) / np.prod(np.array(img_dims))
                bpd += extra
                Metric.append(bpd)
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
    
class Evaluator_EBM_GMM():
    """
    evaluator for a conjugate-EBM with GMM prior
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
    
    def similarity_ebm_density_space(self, train_batch_size, test_batch_size):
        """
        For each test data point, compute the L2 norm distance from the training data in latent space,
        return the confusion matrix
        """
        zs_test, ys_test = self.extract_features(train=False, shuffle=False)
        zs_test = torch.Tensor(zs_test)
        ys_test = torch.Tensor(ys_test)
        pred_ys_test = []
        train_data, _ = load_data(self.dataset, self.data_dir, train_batch_size, train=True, normalize=True, shuffle=False)
        ys_train = torch.Tensor(train_data.dataset.targets)
        for b in range(int(len(ys_test) / test_batch_size)):
            zs_test_b = zs_test[(b)*test_batch_size:(b+1)*test_batch_size]
            zs_test_b = zs_test_b.unsqueeze(1).repeat(1, train_batch_size, 1)
            densities = []
            zs_test_b = zs_test_b.cuda().to(self.device)
            for i, (train_images, train_labels) in enumerate(train_data):
                train_images = train_images.cuda().to(self.device)
                log_prior = self.ebm.log_prior(zs_test_b) # test_size * train_size
                ll = self.ebm.log_factor_expand(train_images, zs_test_b, test_batch_size)
                log_joint = (ll + log_prior).cpu().detach()
                densities.append(log_joint)
            indices = torch.argmax(torch.cat(densities, 1), dim=-1) # test_size
            pred_ys_test.append(ys_train[indices].unsqueeze(-1))
            print('%d completed..' % (b+1))
        pred_ys_test = torch.cat(pred_ys_test, 0)
        return ys_test, pred_ys_test
    
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
    
    def cond_sampling(self, class_label, batch_size, sgld_steps, sgld_lr, sgld_noise_std, grad_clipping=False, init_samples=None, logging_interval=None):
        print('sample conditionally with class=%d..' % class_label)
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
        
        images_ebm = sgld_sampler.sample_cond(class_label=class_label,
                                         ebm=self.ebm, 
                                         batch_size=batch_size, 
                                         num_steps=sgld_steps,
                                         pcd=False,
                                         init_samples=init_samples,
                                         logging_interval=logging_interval)
        if init_samples is not None:
            images_ebm = torch.cat((init_samples.unsqueeze(0), images_ebm), 0)
        return images_ebm
    
    
    def uncond_sampling_ll(self, sample_size, batch_size, sgld_steps, sgld_lr, sgld_noise_std, init_images, grad_clipping=False, logging_interval=None):
        print('encoding initial images..')
        init_images = init_images.cuda().to(self.device)
        init_images = init_images + self.data_noise_std * torch.randn_like(init_images)
        neural_ss1, neural_ss2 = self.ebm.forward(init_images)
        mean_z, _ = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
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
        list_images = []
        list_images.append(init_images)
        for i in range(sample_size):
            images_ebm = sgld_sampler.sample_ll(ebm=self.ebm, 
                                             latents=mean_z,
                                             batch_size=batch_size, 
                                             num_steps=sgld_steps,
                                             init_samples=None,
                                             logging_interval=logging_interval)
            list_images.append(images_ebm)
#             images_ebm = images_ebm + self.data_noise_std * torch.randn_like(init_images)
#             neural_ss1, neural_ss2 = self.ebm.forward(images_ebm)
#             mean_z, _ = nats_to_params(self.ebm.prior_nat1+neural_ss1, self.ebm.prior_nat2+neural_ss2)
        return list_images
    
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
        num_cols = len(list_images)
        num_rows = list_images[0].shape[0]
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0, hspace=0)
        fig = plt.figure(figsize=(fs*num_cols, fs*num_rows))
        
        for j, images in enumerate(list_images):
            images = images.squeeze().cpu().detach()
            images = torch.clamp(images, min=-1, max=1)
            images = images * 0.5 + 0.5
            for i in range(len(images)):
#                 print(i)
                ax = fig.add_subplot(gs[i, j])
                try:
                    ax.imshow(images[i], cmap='gray', vmin=0, vmax=1.0)
                except:
                    ax.imshow(np.transpose(images[i].numpy(), (1,2,0)), vmin=0, vmax=1.0)
                ax.set_axis_off()
                ax.set_xticks([])
                ax.set_yticks([])
            if save:
                plt.savefig('samples/%s_ll_sampling.png' % (self.dataset), dpi=300)
                plt.close()
            
    def extract_energy(self, dataset, train):
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
    
    def extract_features(self, train, shuffle=False):
        test_data, img_dims = load_data(self.dataset, self.data_dir, 1000, train=train, normalize=True, shuffle=shuffle)
        zs = []
        ys = []
        for (images, labels) in test_data:
            images = images.cuda().to(self.device)
#             images = images + self.data_noise_std * torch.randn_like(images)
            neural_ss1, neural_ss2 = self.ebm.forward(images)
            pred_y = self.ebm.posterior_y(images)
            prior_nat1, prior_nat2 = params_to_nats(self.ebm.prior_mu, self.ebm.prior_log_sigma.exp())
            mean_1tk, _ = nats_to_params(prior_nat1.unsqueeze(0)+neural_ss1.unsqueeze(1), prior_nat2.unsqueeze(0)+neural_ss2.unsqueeze(1))
            pred_y_expand = pred_y.argmax(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mean_1tk.shape[2])
            mean = torch.gather(mean_1tk, 1, pred_y_expand).squeeze(1)
#             mean = (pred_y.unsqueeze(2) * mean_1tk).sum(1)
#             mean = mean_1tk.mean(1)
            zs.append(mean.cpu().detach().numpy())
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
#         images = images + self.data_noise_std * torch.randn_like(images)
        neural_ss1, neural_ss2 = self.ebm.forward(images)
        pred_y = self.ebm.posterior_y(images)
        prior_nat1, prior_nat2 = params_to_nats(self.ebm.prior_mu, self.ebm.prior_log_sigma.exp())
        mean_1tk, _ = nats_to_params(prior_nat1.unsqueeze(0)+neural_ss1.unsqueeze(1), prior_nat2.unsqueeze(0)+neural_ss2.unsqueeze(1))
        pred_y_expand = pred_y.argmax(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mean_1tk.shape[2])
        mean = torch.gather(mean_1tk, 1, pred_y_expand).squeeze(1)
#         mean = (pred_y.unsqueeze(2) * mean_1tk).sum(1)
#         mean = mean_1tk.mean(1)
        zs.append(mean.cpu().detach().numpy())
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
        fig = plt.figure(figsize=(fs*2,fs))
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.hist(-energy_id, bins=100, label='in-dist (%s)' % self.dataset, alpha=.3, density=density)
        ax.set_title('Train on %s' % self.dataset, fontsize=14)
        _ = ax.hist(-energy_ood, bins=100, label='out-of-dist (%s)' % dataset , alpha=.3, density=density)
        ax.legend(fontsize=14)
        ax.set_xlabel('- E(x)')
        if save:
            plt.savefig('CEBM_OOD_in=%s_out=%s.png' % (self.dataset, dataset), dpi=300)    