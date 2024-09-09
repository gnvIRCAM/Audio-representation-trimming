
import math 
import os 
import typing as tp 

import gin 
import torch 
import torch.nn as nn 
import torch.utils 
import torch.utils.data

from .utils import gather_masks

class SparsityLoss(nn.Module):
    def __init__(self, 
                 target: float = 0, 
                 power: int = 1, 
                 scale: tp.Callable[[torch.Tensor], torch.Tensor]=torch.sigmoid):
        super().__init__()
        self.scale=scale
        self.target=target 
        self.power=power
        
    def forward(self, masked_model: nn.Module)-> float:
        masks = gather_masks(masked_model)
        if not len(masks.keys()):
            return torch.tensor(0.)
        loss=0
        for mask in masks.values():
            loss=loss+(torch.mean(self.scale(mask))/len(masks))
        loss = (loss-self.target)**(self.power)
        return loss

@gin.configurable(module='train')
def make_sparsity_loss(weight, device, target=0., power=1, scale=torch.sigmoid):
    loss_fn = SparsityLoss(target=target, scale=scale, power=power).to(device)
    def train_callback(pred, labels, model):
        return weight*loss_fn(model)
    return train_callback

@gin.configurable(module='train')
def make_cross_entropy_loss(weight, device):
    loss_fn = nn.CrossEntropyLoss().to(device)
    def train_callback(pred, labels, model):
        return weight*loss_fn(pred, labels)
    return train_callback

@gin.configurable(module='train')
def make_bce_loss(weight, device):
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(device)
    def train_callback(pred, labels, model):
        return weight*loss_fn(pred, labels)
    return train_callback

@gin.configurable(module='train')
def make_ctc_loss(weight, device, blank, pad_token):
    loss_fn = nn.CTCLoss(blank=blank).to(device)
    def train_callback(pred: torch.Tensor, labels: torch.Tensor, model):
        pred = pred.permute(2, 0, 1) # T x B x N
        probs = pred.log_softmax(-1)
        input_lengths = torch.where(pred==pad_token, 0, 1).sum(-1) 
        output_lengths = torch.tensor(probs.shape[0]).unsqueeze(0).repeat(probs.shape[1])
        return weight*loss_fn(probs, labels, input_lengths, output_lengths)
    return train_callback

