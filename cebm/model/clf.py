import torch.nn as nn
from cebm.net import cnn_mlp_1out

class SEMI_CLF(nn.Module):
    """
    a (semi-)supervised classifier
    """
    def __init__(self, **kwargs):
        super().__init__() 
        self.clf_net = nn.Sequential(*(list(cnn_mlp_1out(**kwargs)) + [nn.LogSoftmax(dim=-1)]))
        self.nllloss = nn.NLLLoss()
        
    def pred_logit(self, images):
        return self.clf_net(images)
    
    def loss(self, pred_logits, labels):
        return self.nllloss(pred_logits, labels)
    
    def score(self, images, labels):
        pred_logits = self.pred_logit(images)
        return (pred_logits.argmax(-1) == labels).float().sum()