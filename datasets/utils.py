import json
import os
import typing as tp 

import gin
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import trange

from .dataset import *

@gin.configurable(module='data')
class Tokenizer:
    def __init__(self)-> None:
        self.space_char = '|'
        self.blank_char="-"
        self.characters = list(f"{self.blank_char}{self.space_char}'abcdefghijklmnopqrstuvwxyz".upper())
        self.tokens = {char: i for i, char in enumerate(self.characters)}
        self.reverse_tokens = {v:k for k, v in self.tokens.items()}
        
    def encode_single(self, x: str) -> torch.Tensor:
        if not len(x):
            return torch.tensor([])
        x = x.replace(' ', self.space_char)
        _x_tokens = [self.tokens[char] for char in x]
        x_tokenized = []
        for t, t_next in zip(_x_tokens, _x_tokens[1:]):
            x_tokenized.append(t)
            if t==t_next:
                x_tokenized.append(self.tokens[self.blank_char])
        x_tokenized.append(_x_tokens[-1])
        return torch.tensor(x_tokenized)
    
    def encode_multiple(self, x: tp.List[str])-> torch.Tensor:
        x_tokenized = []
        for text in x:
            x_tokenized.append(self.encode_single(text))
        x_tokenized = pad_sequence(x_tokenized, batch_first=True, 
                                   padding_value=self.tokens[self.blank_char])
        return torch.cat(x_tokenized, dim=0)
    
    def decode_single(self, x: torch.Tensor) -> str:
        _x_reverse = [self.reverse_tokens[t.item()] for t in x]
        transcription = ''
        for char, char_next in zip(_x_reverse, _x_reverse[1:]):
            if char==char_next:
                continue
            transcription+=char
        if _x_reverse[-1]!=_x_reverse[-2]:
            transcription+=_x_reverse[-1]
        transcription = transcription.replace(self.space_char, ' ')
        transcription = transcription.replace(self.blank_char, '')
        return transcription

    def decode_multiple(self, x: torch.Tensor)-> tp.List[str]:
        return [self.decode_single(tokens) for tokens in x]

    def pad_tokens(self, tokens: torch.Tensor):
        tokens = pad_sequence(tokens, 
                              batch_first=True, 
                              padding_value=self.tokens[self.blank_char])
        return torch.cat(tokens, dim=0)

def load_metadata(metadata_path: str):
    with open(metadata_path, 'r') as f:
        return json.load(f)

def make_loaders(dataset_path: str, 
                 bs: int, 
                 num_workers: int = 0, 
                 fold: int = 0)-> torch.utils.data.DataLoader:
    dataset = SimpleDataset(
        dataset_path
    )
    with open(os.path.join(dataset_path, 'dataset_metadata.json'), 'r') as f:
        dataset_sr = json.load(f)['dataset_sr']
    assert fold<len(dataset[0]['metadata']['metadata']['fold']), f"Dataset has {len(dataset[0]['metadata']['metadata']['fold'])} folds, but got fold {fold}"
    
    train_indexes, val_indexes, test_indexes = [], [], []
    num_classes = 0
    for i in trange(len(dataset), desc='Building loaders'):
        example_fold = dataset[i]['metadata']['metadata']['fold'][fold]
        label = dataset[i]['metadata']['metadata']['label']
        if isinstance(label, list):
            if np.all([l in [0, 1] for l in label]):
                label = len(label)
            else:
                label = max(label)+1
        else:
            label = label+1
        if label>num_classes:
            num_classes = label
        if example_fold=='train':
            train_indexes.append(i)
        elif example_fold=='val':
            val_indexes.append(i)
        else:
            test_indexes.append(i)
    if len(test_indexes)==0:
        test_indexes=val_indexes
    
    @torch.no_grad()
    def collate_fn(B):
        B_wav = [torch.tensor(x['waveform']).float()for x in B]
        B_wav = pad_sequence(B_wav, batch_first=True, padding_value=0)
        B_label = [torch.tensor(y['metadata']['metadata']['label']) for y in B]
        if B_label[0].dim()==3:
            B_label = pad_sequence(B_label, batch_first=True, padding_value=0)
        else:
            B_label = torch.stack(B_label, dim=0)
        return B_wav, B_label, [x['metadata']['metadata']['class'] for x in B] 
    
    train_set = Subset(dataset, indices=train_indexes)
    val_set = Subset(dataset, indices=val_indexes)
    test_set = Subset(dataset, indices=test_indexes)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=bs, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        shuffle=True
        )
    val_loader = DataLoader(
        val_set, 
        batch_size=bs, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        shuffle=False
        )
    test_loader = DataLoader(
        test_set, 
        batch_size=bs, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        shuffle=False
        )
    
    return train_loader, val_loader, test_loader, dataset_sr, num_classes