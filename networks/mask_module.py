import typing as tp

import gin 
import torch
import torch.nn as nn

def _unsqueeze_as(x: torch.Tensor, target: torch.Tensor
                  ) -> torch.Tensor:
    for _ in range(target.dim()-x.dim()):
        x=x.unsqueeze(-1)
    return x

@gin.configurable(module='models')
class MaskModule(nn.Module):
    def __init__(self, num_features: int, feature_dim: int=1)->None:
        super().__init__()
        self.feature_dim=feature_dim
        self.num_features=num_features
        self.mask = nn.Parameter(torch.ones(1, num_features))
    
    @property
    def binary_mask(self):
        _bin_mask = torch.round(torch.sigmoid(self.mask)) # Quantize mask values
        bin_mask = self.mask + (_bin_mask-self.mask).detach() # Bypass rounding operator during backward
        return bin_mask
    
    @property
    def masked_indexes(self):
        return torch.where(self.binary_mask==0)

    @property
    def num_masked_features(self):
        return self.num_features-self.binary_mask.sum().item()
    
    def set_mask(self, indexes: tp.List[str], keep: bool=False
                 )-> None:
        if keep:
            self.mask.data = torch.zeros_like(self.mask.data)
            self.mask.data[:, indexes]=1.
        else:
            self.mask.data[:, indexes]=0.
    
    def reset_mask(self):
        self.mask.data = torch.ones_like(self.mask.data)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        mask = _unsqueeze_as(self.binary_mask, x)
        mask = mask.transpose(1, self.feature_dim)
        return x*mask
    
@gin.configurable(module='models')
class ScaleShift(nn.Module):
    def __init__(self, num_features: int, feature_dim: int=1)->None:
        super().__init__()
        self.feature_dim=feature_dim
        self.num_features=num_features
        self.scale = nn.Parameter(torch.ones(1, num_features))
        self.shift = nn.Parameter(torch.zeros(1, num_features))
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        scale = _unsqueeze_as(self.scale, x)
        scale = scale.transpose(1, self.feature_dim)
        shift = _unsqueeze_as(self.shift, x)
        shift = shift.transpose(1, self.feature_dim)
        y = x*scale+shift
        return y
