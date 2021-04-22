import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from cebm.data import setup_data_loader
from cebm.utils import load_models


network_args = {'im_height': im_h, 
                'im_width': im_w, 
                'input_channels': im_channels, 
                'channels': eval(args.channels), 
                'kernels': eval(args.kernels), 
                'strides': eval(args.strides), 
                'paddings': eval(args.paddings),
                'activation': args.activation,
                'hidden_dim': eval(args.hidden_dim),
                'latent_dim': args.latent_dim}
model_args = {'optimize_ib': args.optimize_ib,
              'device': device,
              'num_clusters': args.num_clusters}
models = init_models(args.model_name, device, model_args, network_args)
    

class Evaluator():
    def __init__(self, models, model_name, data, data_dir, device):
        super().__init__()
        self.models = models
        self.model_name = model_name
        self.data = data
        self.data_dir = data_dir
        self.device = device
        
    def latent_mode(self):
        if self.model_name in ['IGEBM', 'CEBM', 'CEBM_GMM']:
            return self.models['ebm'].latent_params()
        elif self.model_name in ['VAE', 'VAE_GMM', 'BIGAN', 'BIGAN_GMM']:
            return self.models['enc'].latent_params()
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
        if os.path.exists('results/few_label/'):
            os.makedirs('results/few_label/')
        results = {'Num_Shots': [], 'Mean': [], 'Std': []}
        for num_shots in tqdm(list_num_shots):
            Accuracy = []
            for i in range(num_runs):
                train_loader, im_h, im_w, im_channels = setup_data_loader(data=self.data, 
                                                                          data_dir=self.data_dir, 
                                                                          num_shots=num_shots, 
                                                                          batch_size=batch_size, 
                                                                          train=True, 
                                                                          normalize=False if self.model_name in ['VAE', 'VAE_GMM'] else True, 
                                                                          shuffle=False, 
                                                                          shot_random_seed=None if num_shots==-1 else i)
                zs_train, ys_train = self.encode_dataset(train_loader)
                if classifier == 'logistic':
                    clf = LogisticRegression(random_state=0, 
                                             multi_class='auto', 
                                             solver='liblinear', 
                                             max_iter=10000).fit(zs_train, ys_train)
                else:
                    raise NotImplementedError
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
        pd.DataFrame.from_dict(results).to_csv('results/few_label/%s-%s-%s', index=False)
            print('clf=%s, model=%s, data=%s, num_shots=%d, mean=%.2f, std=%.2f' % (classifier, self.model_name, self.data, 
        fout.close()
    
    def train_logistic_classifier(zs, ys, ):
        accu = lr.score(zs_test, ys_test)
    #     print('mean accuray=%.4f' % accu)
        return accu
 
        