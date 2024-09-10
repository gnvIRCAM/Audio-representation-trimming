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
                 target: float = 0.5, 
                 power: int = 2 
                 ) -> None:
        super().__init__()
        self.target = target 
        self.power = power
        
    def forward(self, masked_model: nn.Module)-> float:
        masks = gather_masks(masked_model)
        if not len(masks.keys()):
            return torch.tensor(0.)
        loss = 0
        for mask in masks.values():
            loss = loss+(torch.mean(torch.sigmoid(mask))/len(masks))
        loss = (loss-self.target)**(self.power)
        return loss

@gin.configurable(module='train')
def make_sparsity_loss(weight, device, target, power):
    loss_fn = SparsityLoss(target=target, power=power).to(device)
    def train_callback(pred, labels, model):
        return weight*loss_fn(model)
    return train_callback, 'Sparsity loss'

@gin.configurable(module='train')
def make_cross_entropy_loss(weight, device):
    loss_fn = nn.CrossEntropyLoss().to(device)
    def train_callback(pred, labels, model):
        return weight*loss_fn(pred, labels)
    return train_callback, 'Cross-Entropy'

@gin.configurable(module='train')
def make_bce_loss(weight, device):
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(device)
    def train_callback(pred, labels, model):
        return weight*loss_fn(pred, labels)
    return train_callback, 'BCE'

@gin.configurable(module='train')
def make_ctc_loss(weight, device, blank, pad_token):
    loss_fn = nn.CTCLoss(blank=blank).to(device)
    def train_callback(pred: torch.Tensor, labels: torch.Tensor, model):
        pred = pred.permute(2, 0, 1) # T x B x N
        probs = pred.log_softmax(-1)
        input_lengths = torch.where(pred==pad_token, 0, 1).sum(-1) 
        output_lengths = torch.tensor(probs.shape[0]).unsqueeze(0).repeat(probs.shape[1])
        return weight*loss_fn(probs, labels, input_lengths, output_lengths)
    return train_callback, 'CTC'

@gin.configurable(module='train')
def make_losses(task_loss, sparsity_loss=None):
    task_loss_fn, task_loss_name = task_loss()
    loss_dict = {task_loss_name: task_loss_fn()}
    if sparsity_loss is not None:
        sparsity_loss_fn, sparsity_loss_name = task_loss()
        loss_dict[sparsity_loss_name]=sparsity_loss_fn()
    return loss_dict