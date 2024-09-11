import typing as tp

from einops import rearrange
import gin
import torch
import torch.nn as nn 

__all__ = ['MLP', 'BLSTM']

class HeadBase(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        
    def get_classification(self, x: torch.Tensor)->torch.Tensor:
        pred = self(x)
        return torch.argmax(pred, dim=1)    
    
@gin.configurable(module='models')
class MLP(HeadBase):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: tp.List[int] = [], 
        hidden_act: nn.Module = nn.ReLU,
        out_act: tp.Optional[nn.Module] = None) -> None:
        super().__init__(output_dim)
        
        self.layers = []
        if not len(hidden_dims):
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for d_in, d_out in zip(hidden_dims, hidden_dims[1:]+[output_dim]):
            self.layers.append(hidden_act())
            self.layers.append(nn.Linear(d_in, d_out))
        if out_act is not None:
            self.layers.append(out_act())
        
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.layers(x)

@gin.configurable(module='models')    
class BLSTM(HeadBase):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 256, 
        num_layers: int = 1, 
        out_act: tp.Optional[nn.Module] = None) -> None:
        super().__init__(output_dim)
           
        self.layers = []
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.proj = nn.Linear(2*hidden_dim, output_dim)
        self.out_act = out_act()

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.proj(x)
        if self.out_act is not None:
            x = self.out_act(x)
        x = x.permute(0, 2, 1) # b t c -> b c t
        return x

                 
        