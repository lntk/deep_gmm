import torch
from torch.nn import functional as F
import numpy as np  
import torch.nn as nn


def negative_log_likelihood_loss(weights, means, targets):
    sequence_length = weights.shape[0]    
    
    x = weights * torch.pow(means, targets.float()) * torch.pow(1 - means, 1.0 - targets.float())    
    losses = -torch.log(weights * torch.pow(means, targets.float()) * torch.pow(1 - means, 1.0 - targets.float())) # S x B x C x H x W        
    
    # losses = torch.mean(losses, dim=0)
    # loss = torch.sum(losses)
    
    loss = torch.mean(losses)    
    
    return loss

# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
def jaccard_loss(pred, true, eps=0):    
    B, S, _, H, W = true.shape    
    
    intersection = pred * true
    union = pred + true - intersection
    
    intersection = torch.sum(intersection, dim=(2, 3, 4))
    union = torch.sum(union, dim=(2, 3, 4))
    
    loss = (intersection + eps) / (union + eps)  # B x S
    inv_loss = 1 - loss
    inv_loss = torch.mean(inv_loss)    
    
    return inv_loss

class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()
    
    def forward(self, pred, true, eps=1e-7):
        B, S, _, H, W = true.shape    
        
        intersection = pred * true
        # union = pred + true - intersection
        
        intersection = torch.sum(intersection, dim=(2, 3, 4))
        # union = torch.sum(union, dim=(2, 3, 4))
                
        # loss = (intersection + eps) / (union + eps)  # B x S    
        loss = (2 * intersection + eps) / (torch.sum(pred, dim=(2, 3, 4)) + torch.sum(true, dim=(2, 3, 4)) + eps)  # B x S    
        
        inv_loss = 1 - loss
        inv_loss = torch.mean(inv_loss)    
        
        return inv_loss        
        

def binary_cross_entropy(output, target):
    return F.binary_cross_entropy(output, target)


def mean_iou_score(pred, true, epsilon=1e-9):
    """
    
    @pred: H x W
    """                         
    
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    
    intersection = np.logical_and(pred, true).astype("uint8")
    overlap = np.logical_or(pred, true).astype("uint8")
    
    intersection = np.sum(intersection)
    overlap = np.sum(overlap)
    
    score = (intersection + epsilon) / (overlap + epsilon)
    
    return score
    