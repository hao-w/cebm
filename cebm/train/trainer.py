import os
import torch
from tqdm import tqdm
from cebm.utils import save_models

class Trainer():
    """
    A generic model trainer
    """
    def __init__(self, models, train_loader, num_epochs, device, exp_name):
        super().__init__()
        self.models = models
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.exp_name = exp_name
        
    def train_epoch(self, epoch):
        pass
    
    def train(self):
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:            
            metric_epoch = self.train_epoch(epoch)
            pbar.set_postfix(ordered_dict=metric_epoch)
            self.logging(metric_epoch, epoch)
            save_models(self.models, self.exp_name)
    
    #FIXME: hao will replace this function with tensorboard API later.
    def logging(self, metrics, epoch):
        if not os.path.exists('./logging/'):
            os.makedirs('./logging/')
        fout = open('./logging/' + self.exp_name + '.txt', mode='w+' if epoch==0 else 'a+')
        metric_print = ",  ".join(['%s=%.2e' % (k, v) for k, v in metrics.items()])
        print("Epoch=%d, " % (epoch+1) + metric_print, file=fout)
        fout.close()