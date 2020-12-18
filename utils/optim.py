import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
import torch.nn.functional as F

class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

def linear_combination(x, y, epsilon): 
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)
