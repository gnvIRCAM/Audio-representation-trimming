import csv
from collections import defaultdict
from itertools import chain
import json
import os 
from pathlib import Path
import typing as tp 
import yaml

import gin
import torch
from torch.nn.utils.rnn import pad_sequence 

@gin.configurable(module='data')
class Tokenizer:
    def __init__(self)-> None:
        self.space_char = '|'
        self.blank_char="-"
        self.characters = list(f"{self.blank_char}{self.space_char}'abcdefghijklmnopqrstuvwxyz".upper())
        self.tokens = {char: i for i, char in enumerate(self.characters)}
        self.reverse_tokens = {v:k for k, v in self.tokens.items()}
        
    def encode_single(self, x: str) -> torch.Tensor:
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

# Get rid of those and preparse metadata by hand and include json files
def get_metadata_nsynth(nsynth_path: str, fold: int = 0):
    metadata = {}
    assert fold==0, 'Nsynth has only a single cross-validation split'
    train, val, test = [], [], []
    train_path = os.path.join(nsynth_path, 'nsynth-train', 'examples.json')
    eval_path = os.path.join(nsynth_path, 'nsynth-valid', 'examples.json')
    test_path = os.path.join(nsynth_path, 'nsynth-test', 'examples.json')
    for meta_path in [train_path, eval_path, test_path]:
        with open(os.path.join(nsynth_path, meta_path), 'r') as f:
            labels = json.load(f)
            for k, v in labels.items():
                file_path = os.path.join(nsynth_path, 'audio', k+'.wav')
                metadata[file_path]={
                    'class' : v['pitch'], 
                    'label' : v['pitch'], 
                }
                if meta_path==train_path:
                    train.append(file_path)
                if meta_path==eval_path:
                    val.append(file_path)
                if meta_path==test_path:
                    test.append(file_path)
    return metadata, train, val, test

def get_metadata_gtzan(gtzan_path: str, fold: int = 0):
    assert fold in [0, 1, 2], f'GTZAN has only 3 cross-validation folds, but got fold {fold}'
    metadata = {}
    p = Path(gtzan_path)
    genres = []
    for x in p.rglob('*.au'):
        genre = p.split('/')[-1].split('.')[0]
        genres.append(genre)
        metadata[x]= {
            'class': genre
        }
    genres = list(set(genres))
    labels = {genre: i for i, genre in enumerate(sorted(genres))}
    for path, cls in metadata.items():
        metadata[path]['label'] = labels[genre]

    train_fold = os.path.join('data/gtzan_folds', f'f{fold+1}_train.txt')
    val_fold = os.path.join('data/gtzan_folds', f'f{fold+1}_evaluate.txt') 
    test_fold = os.path.join('data/gtzan_folds', f'f{fold+1}_test.txt')
    
    train, val, test = [], [], []    
    
    def _format_filename(filename):
        filename = filename.replace('./', '')
        filename = filename.replace('.wav', 'au')
        filename = filename.replace('_', '.')
        return list(p.rglob(filename))
    
    with open(train_fold, 'r') as f:
        fold = f.read().split('\n')
        file = fold.split(' ')[0]
        train.append(_format_filename(file))     
    with open(val_fold, 'r') as f:
        fold = f.read().split('\n')
        file = fold.split(' ')[0] 
        val.append(_format_filename(file))     
    with open(test_fold, 'r') as f:
        fold = f.read().split('\n')
        file = fold.split(' ')[0]
        test.append(_format_filename(file))     
    
    return metadata, train, val, test 

def get_metadata_mtt(mtt_path: str, fold: int = 0):
    pass

def get_metadata_esc():
    pass

def get_metadata_us8k():
    pass

def get_metadata_fsd():
    pass

def get_metadata_libri():
    pass

def get_metadata_fluent():
    pass

def get_metadata_commands():
    pass