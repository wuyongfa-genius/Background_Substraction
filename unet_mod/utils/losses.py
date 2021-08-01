import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels):
        logits = logits.squeeze(1)
        probs = logits.sigmoid() # BHW
        valid_mask = labels!=self.ignore_index
        labels = labels*(valid_mask)
        labels = labels.type_as(logits)
        pt = probs*labels+(1-probs)*(1-labels)
        alphat = self.alpha*labels+(1-self.alpha)*(1-labels)
        focal_weight = alphat*(1-pt)**(self.gamma)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')*focal_weight
        return torch.sum(loss*(valid_mask))/float(torch.sum(valid_mask))


class FocalLoss(nn.Module):
    """
    Focal Loss that supports multi-class classification.
    Args:
        alpha(float | torch.Tensor): the gt class 's weight is alpha,
            and other classes's weights are all 1-alpha.
        gamma(float): modulating factor to down-weight the 
            correctly-classified samples.
    """
    def __init__(self, alpha=1., gamma=2., ignore_index=255):
        super().__init__()
        assert alpha>0 and alpha<=1
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels):
        """
        Args:
            logits(torch.Tensor): logits of shape BCHW
            labels(torch.LongTensor): labels of shape BHW
        """
        C = logits.shape[1]
        valid_mask = labels!=self.ignore_index
        labels = labels*(valid_mask) # BHW
        one_hot_labels = F.one_hot(labels, C)
        one_hot_labels = rearrange(one_hot_labels, 'b h w c -> b c h w').contiguous()
        one_hot_labels = one_hot_labels.type_as(logits)
        probs = F.softmax(logits, 1)
        pt = probs*one_hot_labels+(1-probs)*(1-one_hot_labels)
        alphat = self.alpha*one_hot_labels+(1-self.alpha)*(1-one_hot_labels)
        focal_weight = alphat*(1-pt)**(self.gamma)
        cross_entropy = -(focal_weight*one_hot_labels*F.log_softmax(logits, 1)).sum(1) # BHW
        return torch.sum(cross_entropy*valid_mask)/float(torch.sum(valid_mask)) #cross_entropy.mean()

       
# if __name__=="__main__":
#     logits = torch.zeros((1, 3, 1, 1))
#     logits[0,:,0,0] = torch.tensor([0.1, 0.99, 0.1])
#     labels = torch.zeros((1, 1, 1), dtype=torch.long)
#     labels[0,0,0] = 1
#     criterion1 = FocalLoss(alpha=0.25, gamma=2.)
#     loss1 = criterion1(logits, labels)
#     criterion2 = nn.CrossEntropyLoss()
#     loss2 = criterion2(logits, labels)
#     print(loss1)
#     print(loss2)
        
            