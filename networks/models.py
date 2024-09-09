import os 
import typing as tp

import gin
import torch 
import torch.nn as nn
import torchaudio

from msclap import CLAP
from musicfm.model.musicfm_25hz import MusicFM25Hz

__all__ = ['End2EndCLAP', 'End2EndMusicFM', 'End2EndWav2Vec']

class End2EndModel(nn.Module):
    """
    Base class to build an end-to-end model by 
    assembling a foundation model and a head
    """
    def __init__(self, 
                 foundation_model: nn.Module, 
                 head: nn.Module)->None:
        super().__init__()
        self.foundation_model = foundation_model
        self.head = head
    
    @property
    def num_classes(self):
        return self.head.num_classes 

    def train(self):
        self.foundation_model.eval()
        self.head.train()
    
    def eval(self):
        self.foundation_model.eval()
        self.head.eval()
    
    def get_embedding(self, x: torch.Tensor)->torch.Tensor:
        """
        Given an input x, will return the embedding of x using the foundation model
        """
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor):
        embedding = self.get_embedding(x)
        return self.head(embedding)

@gin.configurable(module='models')
class End2EndCLAP(End2EndModel):
    def __init__(self, clap: nn.Module, head: nn.Module) -> None:
        super().__init__(clap, head)

    def get_embedding(self, x: torch.Tensor)->torch.Tensor:
        emb, _ = self.foundation_model.audio_encoder(x)
        return emb

@gin.configurable(module='models')
class End2EndMusicFM(End2EndModel):
    def __init__(self, 
                 musicfm: nn.Module,
                 head: nn.Module, 
                 layer_idx: int=-1, 
                 time_avg: bool=True) -> None:
        super().__init__(musicfm, head)
        self.layer_idx = layer_idx
        self.time_avg = time_avg
        
    def get_embedding(self, x: torch.Tensor)->torch.Tensor:
        emb = self.foundation_model.get_latent(x, layer_ix=self.layer_idx)
        if self.time_avg:
            emb = emb.mean(1) # B T C -> B C
        return emb

@gin.configurable(module='models') 
class End2EndWav2Vec(End2EndModel):
    def __init__(self, 
                 wav2vec: nn.Module, 
                 head: nn.Module, 
                 layer_idx: int=-1, 
                 time_avg: bool=True) -> None:
        super().__init__(wav2vec, head)
        self.layer_idx = layer_idx
        self.time_avg = time_avg

    def get_embedding(self, x: torch.Tensor)->torch.Tensor:
        emb = self.foundation_model.extract_features(x)[self.layer_idx]
        if self.time_avg:
            emb = emb.mean(1) # B T C -> B C
        return emb