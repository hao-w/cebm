import torch
import argparse
from cebm.data import setup_data_loader
from cebm.utils import set_seed, create_exp_name, init_models
from cebm.train.trainer import Trainer

class Train_SEMI_CLF(Trainer):
    def __init__(self, models, train_loader, num_epochs, device, exp_name, optimizer, lr, reg_lambda, test_loader):
        super().__init__(models, train_loader, num_epochs, device, exp_name)
        self.optimizer = getattr(torch.optim, optimizer)(list(self.models['clf'].parameters()), lr=lr, betas=(0.5, 0.999))
        self.reg_lambda = reg_lambda
        self.metric_names = ['accuracy']
        self.test_loader = test_loader
        
    def test_score(self):
        accu = 0.0
        N = 0.0
        for b, (images, labels) in enumerate(self.test_loader): 
            N += len(images)
            images = images.to(self.device)
            labels = labels.to(self.device)
            accu += self.models['clf'].score(images, labels).detach()
        return accu / N
    
    def train_epoch(self, epoch):
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, (images, labels) in enumerate(self.train_loader): 
            self.optimizer.zero_grad()
            images = images.to(self.device)
            labels = labels.to(self.device)
            pred_logits = self.models['clf'].pred_logit(images)
            # compute cross entropy loss for labeled data
            labeled_labels = labels[(labels != -1)]
            if labeled_labels.shape != (0,):
                labeled_logits = pred_logits[(labels != -1)]
                loss = self.models['clf'].loss(labeled_logits, labeled_labels)
            else:
                loss = 0.0
            # compute entropy loss for unlabeled data
            if self.reg_lambda != 0:
                unlabeled_labels = labels[(labels == -1)]
                if unlabeled_labels.shape != (0,):
                    unlabeled_logits = pred_logits[(labels == -1)]
                    loss += self.reg_lambda * (- (unlabeled_logits.exp() * unlabeled_logits).sum(-1)).sum()
            try:
                loss.backward()
                self.optimizer.step()
            except:
                continue
        metric_epoch['accuracy'] = self.test_score()
        return {k: v.item() for k, v in metric_epoch.items()} 

def init_models(model_name, device, network_args):
    if model_name == 'SEMI_CLF':
        clf =  SEMI_CLF(**network_args)
    return {'clf': clf.to(device)}

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    
    train_set_args = {'data': args.data, 
                      'data_dir': args.data_dir, 
                      'num_shots': args.num_shots,
                      'batch_size': args.batch_size,
                      'train': True, 
                      'normalize': False,
                      'seed': args.seed}
    test_set_args = {'data': args.data, 
                     'data_dir': args.data_dir, 
                     'num_shots': -1,
                     'batch_size': args.batch_size,
                     'train': False, 
                     'normalize': False}  
    train_loader, im_h, im_w, im_channels = setup_data_loader(**train_set_args)
    test_loader, _, _, _ = setup_data_loader(**test_set_args)
    
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
    models = init_models(args.model_name, device, network_args)
    exp_name = create_exp_name(args)
    print("Experiment: %s" % exp_name)

    trainer_args = {'models': models,
                    'train_loader': train_loader,
                    'num_epochs': args.num_epochs,
                    'device': device,
                    'exp_name': exp_name,
                    'optimizer': args.optimizer,
                    'lr': args.lr,
                    'reg_lambda': args.reg_lambda,
                    'test_loader': test_loader}
    trainer = Train_SEMI_CLF(**trainer_args)
    print('Start Training..')
    trainer.train()  
    
def parse_args():
    parser = argparse.ArgumentParser('Semi-Supervised Classifier')
    parser.add_argument('--model_name', required=True, choices=['SEMI_CLF'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--exp_id', default=None)
    ## data config
    parser.add_argument('--data', required=True, choices=['mnist', 'cifar10', 'svhn', 'fashionmnist'])
    parser.add_argument('--data_dir', default='../datasets/', type=str)
    ## optim config
    parser.add_argument('--optimizer', choices=['AdamW', 'Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    ## arch config
    parser.add_argument('--channels', default="[64,64,32,32]")
    parser.add_argument('--kernels', default="[3,4,4,4]")
    parser.add_argument('--strides', default="[1,2,2,2]")
    parser.add_argument('--paddings', default="[1,1,1,1]")
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--activation', default='ReLU')
    ## training config
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    ## semi-supervised learning config
    parser.add_argument('--num_shots', default=1, type=int)
    parser.add_argument('--reg_lambda', default=0.0, type=float)
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)