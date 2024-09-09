from copy import deepcopy
import typing as tp 
from types import MethodType

import torch 
import torch.nn as nn 

import nn_trim
from .utils import *

# Remove nn_trim from here

def trim_clap(clap: nn.Module)->nn.Module:
    """
    Given a CLAP model trained with binary masks, will convert the masks 
    into trimming indexes, then will trim the model accordingly
    """
    trimmed_clap = deepcopy(clap)
    wrapper = nn_trim.NNTrimmer(trimmed_clap)
    _name = 'network.0.audio_encoder.base'

    for block in [1, 2, 3, 4, 5, 6]:
        for conv in [1, 2]:
            cur_conv_name = _name+f'.conv_block{block}.conv{conv}'
            cur_bn_name = _name+f'.conv_block{block}.bn{conv}'
            if conv==1:
                next_layer_name = _name+f'.conv_block{block}.conv{conv+1}'
            else:
                if block<6:
                    next_layer_name = _name+f'.conv_block{block+1}.conv{conv-1}'
                else:
                    next_layer_name = _name+f'.fc1'
            wrapper.declare_next_layers(cur_conv_name, next_layer_name)
            wrapper.bind_layer_norm(cur_conv_name, cur_bn_name)
            if nn_trim.sequential_hasattr(trimmed_clap, cur_bn_name+'.mask_module'):
                mask = nn_trim.sequential_getattr(trimmed_clap, cur_bn_name+'.mask_module.binary_mask')
                kept_out_nodes = torch.argwhere(mask!=0)[:, 1]
                nn_trim.sequential_setattr(trimmed_clap, cur_conv_name+'.kept_out_nodes', kept_out_nodes)
                nn_trim.sequential_delattr(trimmed_clap, cur_bn_name+'.mask_module')
    wrapper.update()
    for m in trimmed_clap.modules():
        if hasattr(m, 'weight'):
            nn_trim.trim_weight_bias(m)
    remove_all_hooks(trimmed_clap)
    return trimmed_clap
