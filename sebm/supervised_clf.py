import torch
import time
from torchvision import datasets, transforms
from sebm.models import Clf


class Downstream_Semi_Clf():
    def __init__(self, device, reg_lambda, batch_size=100, num_epochs=100):
        super(self.__class__, self).__init__()
        
        self.clf = Clf(arch='mlp', input_dim=128, hidden_dim=[64], latent_dim=10, activation='ReLU').cuda().to(device)   
        self.optimizer = torch.optim.Adam(list(self.clf.parameters()), lr=1e-3)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.reg_lambda = reg_lambda

    def score(self, zs, ys):
        num_batches = int(len(ys) / self.batch_size)
        N = 0.0
        accu = 0.0
        for b in range(num_batches):
            latents = zs[b*self.batch_size:(b+1)*self.batch_size].cuda().to(self.device)
            labels = ys[b*self.batch_size:(b+1)*self.batch_size].cuda().to(self.device)
            N += len(labels)
            log_q = self.clf.forward(latents)
            accu += self.clf.score(log_q, labels).detach()
        accu /= N
        return accu
    
    def train(self, zs, ys):
        for epoch in range(self.num_epochs):
            time_start = time.time()
            metrics = {'loss' : 0.0}
            num_batches = int(len(ys) / self.batch_size)
            indices = torch.randperm(len(ys))
            for b in range(num_batches):
                self.optimizer.zero_grad()
                batch_indices = indices[b*self.batch_size:(b+1)*self.batch_size]
                latents = zs[batch_indices].cuda().to(self.device)
                labels = ys[batch_indices].cuda().to(self.device)
                log_q = self.clf.forward(latents)
                # for labeled data
                labeled_labels = labels[(labels != -1)]
                labeled_log_q = log_q[(labels != -1)]
                if labeled_labels.shape != (0,):
                    cross_entropy = self.clf.loss(labeled_log_q, labeled_labels)
                else:
                    cross_entropy = 0.0
                # for unlabeled data
                unlabeled_labels = labels[(labels == -1)]
                unlabeled_log_q = log_q[(labels == -1)]
                if unlabeled_labels.shape != (0,):
                    entropy = (- (unlabeled_log_q.exp() * unlabeled_log_q).sum(-1)).sum()
                else:
                    entropy = 0.0
                loss = cross_entropy + self.reg_lambda * entropy
                loss.backward()
                self.optimizer.step()
                metrics['loss'] += loss.detach()
#             self.save_checkpoints()
            metrics['loss'] = metrics['loss'] / float(b+1)
#             self.logging(metrics=metrics, epoch=epoch)
            time_end = time.time()
#             print("Epoch=%d / %d completed  in (%ds),  " % (epoch+1, self.num_epochs, time_end - time_start))

    
    
