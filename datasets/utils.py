import json
import typing as tp 

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