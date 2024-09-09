import math
import os
import typing as tp

import gin
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
import torch.utils
import torch.utils.data

def gather_masks(masked_model: nn.Module
                 ) -> tp.Dict[str, nn.Parameter]:
    masks = {}
    for n, m in masked_model.named_modules():
        if hasattr(m, 'mask_module'):
            masks[n] = m.mask_module.mask
    return masks

@gin.configurable(module='train', denylist=['model'])
def make_optimizer(
    model: nn.Module, 
    lr: float, 
    denylist: tp.List[str], 
    exclude_masks: bool = False, 
    include_fm: bool = False
    ) -> Adam:
    params = []
    mask_params = {'name': 'masks', 'params':[]}
    for n, m in model.named_modules():
        if hasattr(m, 'mask_module') and (n not in denylist):
            mask_params['params']+=list(m.mask_module.parameters())
    if len(mask_params) and not exclude_masks:
        params.append(mask_params)
    params.append({'params':list(model.head.parameters()), 
             'name': 'mlp'})
    if include_fm:
        params.append({'params':list(model.foundation_model.parameters()), 
                'name': 'foundation_model'})
    return Adam(params, lr=lr, betas=[.9, .999])


