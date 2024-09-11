import os 
import sys
sys.path.append(os.environ['MUSICFM_REPO_PATH'])
import typing as tp

import gin
import torch 
import torch.nn as nn
import torchaudio

from msclap import CLAP
from musicfm.model.musicfm_25hz import MusicFM25Hz

__all__ = ['combine_fm_and_head']

torch.hub.set_dir(os.environ['TORCH_HUB']) 

class End2EndModel(nn.Module):
    """
    Base class to build an end-to-end model by 
    assembling a foundation model and a head
    """
    def __init__(self, 
                 foundation_model: nn.Module, 
                 head: nn.Module, 
                 model_sr: int, 
                 data_sr: int)->None:
        super().__init__()
        self.foundation_model = foundation_model
        self.head = head
        self.resampler = torchaudio.transforms.Resample(data_sr, model_sr)
    
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
        with torch.no_grad():
            x = self.resampler(x)
        embedding = self.get_embedding(x)
        return self.head(embedding)

@gin.configurable(module='models')
class End2EndCLAP(End2EndModel):
    def __init__(self, 
                 head: nn.Module, 
                 data_sr: int) -> None:
        clap = CLAP(version='2022')
        clap = clap.clap
        delattr(clap, 'caption_encoder') 
        super().__init__(clap, head, model_sr=44100, data_sr=data_sr)

    def get_embedding(self, x: torch.Tensor)->torch.Tensor:
        emb, _ = self.foundation_model.audio_encoder(x)
        return emb

@gin.configurable(module='models')
class End2EndMusicFM(End2EndModel):
    def __init__(self, 
                 head: nn.Module, 
                 data_sr: int, 
                 layer_idx: int=-1, 
                 time_avg: bool=True) -> None:
        musicfm_weights = os.environ['MUSICFM_WEIGHTS_PATH']
        musicfm = MusicFM25Hz(
            is_flash=False, 
            stat_path=os.path.join(musicfm_weights, 'msd_stats.json'),
            model_path=os.path.join(musicfm_weights, 'pretrained_msd.pt'),        
        )
        super().__init__(musicfm, head, model_sr=24000, data_sr=data_sr)
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
                 head: nn.Module, 
                 data_sr: int, 
                 layer_idx: int=-1, 
                 time_avg: bool=True) -> None:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        sr = bundle.sample_rate 
        gin.bind_parameter('%FM_SR', sr)
        wav2vec = bundle.get_model()
        super().__init__(wav2vec, head, model_sr=16000, data_sr=data_sr)
        self.layer_idx = layer_idx
        self.time_avg = time_avg

    def get_embedding(self, x: torch.Tensor)->torch.Tensor:
        emb = self.foundation_model.extract_features(x)[self.layer_idx]
        if self.time_avg:
            emb = emb.mean(1) # B T C -> B C
        return emb

@gin.configurable(module='models')     
def combine_fm_and_head(
    head: nn.Module, 
    end2end_class: End2EndModel, 
    **kwargs) -> End2EndModel:
    return end2end_class(head, **kwargs)