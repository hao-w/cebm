import torch
import time
from torchvision import datasets, transforms
from sebm.models import Clf


class Train_procedure():
    """
    training procedure of the semi-supervised classifier
    """
    def __init__(self, optimizer, clf, num_epochs, batch_size, reg_lambda, device, save_version):
        super(self.__class__, self).__init__()
        self.optimizer = optimizer
        self.clf = clf  
#         self.train_data = train_data
#         self.train_label = train_label
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.reg_lambda = reg_lambda
        self.device = device
        self.save_version = save_version
    def test_score(self, test_data, test_label):
        num_batches = int(len(test_label) / self.batch_size)
        N = 0.0
        accu = 0.0
        for b in range(num_batches):
            x = test_data[b*self.batch_size:(b+1)*self.batch_size].cuda().to(self.device)
            y = test_label[b*self.batch_size:(b+1)*self.batch_size].cuda().to(self.device)
            N += len(y)
            log_q = self.clf.forward(x)
            accu += self.clf.score(log_q, y).detach()
        accu /= N
        return accu
    
    def train(self, train_data, train_label, test_data, test_label):
        for epoch in range(self.num_epochs):
            time_start = time.time()
            metrics = {'loss' : 0.0}
            num_batches = int(len(train_label) / self.batch_size)
            indices = torch.randperm(len(train_label))
            N = 0.0
            for b in range(num_batches):
                self.optimizer.zero_grad()
                batch_indices = indices[b*self.batch_size:(b+1)*self.batch_size]
                x = train_data[batch_indices].cuda().to(self.device)
                y = train_label[batch_indices].cuda().to(self.device)
                log_q = self.clf.forward(x)
                # for labeled data
                labeled_labels = y[(y != -1)]
                labeled_log_q = log_q[(y != -1)]
                if labeled_labels.shape != (0,):
                    N += 1
#                     cross_entropy = self.clf.loss(labeled_log_q, labeled_labels)
                    loss = self.clf.loss(labeled_log_q, labeled_labels)
                    loss.backward()
                    self.optimizer.step()
                    metrics['loss'] += loss.detach()
#                 else:
#                     cross_entropy = 0.0
                # for unlabeled data
#                 unlabeled_labels = y[(y == -1)]
#                 unlabeled_log_q = log_q[(y == -1)]
#                 if unlabeled_labels.shape != (0,):
#                     entropy = (- (unlabeled_log_q.exp() * unlabeled_log_q).sum(-1)).sum()
#                 else:
#                     entropy = 0.0
#                 loss = cross_entropy + self.reg_lambda * entropy
                
                
            metrics['loss'] /= N
            metrics['score'] = self.test_score(test_data, test_label)
            self.save_checkpoints()
            self.logging(metrics=metrics, epoch=epoch)
            time_end = time.time()
            print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))
            
    def logging(self, metrics, epoch):
        if epoch == 0:
            log_file = open('results/log-' + self.save_version + '.txt', 'w+')
        else:
            log_file = open('results/log-' + self.save_version + '.txt', 'a+')
        metrics_print = ",  ".join(['%s=%.3e' % (k, v) for k, v in metrics.items()])
        print("Epoch=%d, " % (epoch+1) + metrics_print, file=log_file)
        log_file.close()
        
    def save_checkpoints(self):
        checkpoint_dict  = {
            'model_state_dict': self.clf.state_dict()
            }
        torch.save(checkpoint_dict, "weights/cp-%s" % self.save_version)


if __name__ == "__main__":
    import torch
    import argparse
    from util import set_seed
    from sebm.data import load_data_remove_labels, load_data_as_array
    parser = argparse.ArgumentParser('Semi-Supervised Classifier')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)
    ## data config
    parser.add_argument('--dataset', required=True, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet', 'celeba', 'flowers102', 'fashionmnist'])
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    ## optim config
    parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    ## arch config
    parser.add_argument('--arch', default='simplenet', choices=['simplenet', 'mlp'])
    parser.add_argument('--channels', default="[64, 64, 32, 32]")
    parser.add_argument('--kernels', default="[3, 4, 4, 4]")
    parser.add_argument('--strides', default="[1, 2, 2, 2]")
    parser.add_argument('--paddings', default="[1, 1, 1, 1]")
    parser.add_argument('--hidden_dim', default="[128]")
    parser.add_argument('--latent_dim', default=10, type=int)
    parser.add_argument('--activation', default='ReLU')
    ## training config
    parser.add_argument('--num_epochs', default=100, type=int)
    ## semi-supervised learning config
    parser.add_argument('--num_shots', default=-1, type=int)
    parser.add_argument('--reg_lambda', default=1e-2, type=float)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device('cuda:%d' % args.device)
    save_version ='semi_clf-data=%s-shots=%s-seed=%d-lr=%s' % (args.dataset, args.num_shots, args.seed, args.lr)
    print('Experiment with ' + save_version)
    print('Loading dataset=%s...' % args.dataset)
    train_data, train_label = load_data_remove_labels(args.dataset, args.data_dir, args.num_shots)
    train_data = torch.Tensor(train_data)
    train_label = torch.LongTensor(train_label)
    
    test_data, test_label = load_data_as_array(args.dataset, args.data_dir, train=False, normalize=False, flatten=False)
    test_data = torch.Tensor(test_data)
    test_label = torch.LongTensor(test_label)
    
    (input_channels, im_height, im_width) = train_data.shape[1:]     
    clf = Clf(arch=args.arch,
                im_height=im_height, 
                im_width=im_width, 
                input_channels=input_channels, 
                channels=eval(args.channels), 
                kernels=eval(args.kernels), 
                strides=eval(args.strides), 
                paddings=eval(args.paddings), 
                hidden_dim=eval(args.hidden_dim),
                latent_dim=args.latent_dim,
                activation=args.activation)
    
    clf = clf.cuda().to(device)
    optimizer = getattr(torch.optim, args.optimizer)(list(clf.parameters()), lr=args.lr)
    print('Start training...')
    trainer = Train_procedure(optimizer, clf, args.num_epochs, args.batch_size, args.reg_lambda, device, save_version)
    trainer.train(train_data, train_label, test_data, test_label)
