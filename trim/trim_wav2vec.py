import typing as tp 
from types import MethodType

import torch 
import torch.nn as nn 

from .utils import *
    
def _wav2vec_mha_layertrim_forward(self,
                            x,
                            attention_mask = None,
                            position_bias = None,key_padding_mask = None):
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        bias = self.out_bias
        if bias is None:
            return torch.zeros_like(x), None
        bias: torch.Tensor = bias.data.unsqueeze(0).unsqueeze(0)
        output = bias.repeat(batch_size, length, 1) # B, L, E
        return output, None
    
def _wav2vec_ffn_layertrim_forward(self, x):
    x = torch.zeros_like(x)
    if self.out_bias is not None:
        bias = self.out_bias.unsqueeze(0).unsqueeze(0)
        x += bias
    x = self.output_dropout(x)
    return x

def _handle_attn_layer_trimming(attention_module: nn.Module)-> nn.Module:
    attention_module.out_bias = attention_module.out_proj.bias
    delattr(attention_module, 'q_proj')
    delattr(attention_module, 'k_proj')
    delattr(attention_module, 'v_proj')
    delattr(attention_module, 'out_proj')
    attention_module.forward = MethodType(_wav2vec_mha_layertrim_forward, attention_module)
    return attention_module

def _handle_ffn_layer_trimming(ffn_module: nn.Module)->nn.Module:
    ffn_module.out_bias = ffn_module.output_dense.bias
    delattr(ffn_module, 'intermediate_dense')
    delattr(ffn_module, 'intermediate_dropout')
    delattr(ffn_module, 'output_dense')
    ffn_module.forward = MethodType(_wav2vec_ffn_layertrim_forward, ffn_module)
    return ffn_module

def _trim_attention(attention_module: nn.Module
                   )->nn.Module:
    if not hasattr(attention_module, 'mask_module'):
        return attention_module
    mask = attention_module.mask_module.binary_mask.squeeze()
    if torch.sum(mask).item()==0:
        print('Attn layer Trimming')
        return _handle_attn_layer_trimming(attention_module)
    kept_heads = [x[0] for x in torch.argwhere(mask==1).tolist()]
    num_heads = len(mask)
    remaining_heads = torch.count_nonzero(mask)
    embed_dim = attention_module.q_proj.weight.shape[0]
    head_size = embed_dim//num_heads
    
    def compute_head_indexes(head: int):
        return [(head*head_size)+idx for idx in range(head_size)]
    kept_indexes = []
    
    for m in kept_heads:
        kept_indexes+=compute_head_indexes(m)
        
    attention_module.q_proj.kept_out_nodes=kept_indexes
    trim_weight_bias(attention_module.q_proj)
    attention_module.k_proj.kept_out_nodes=kept_indexes
    trim_weight_bias(attention_module.k_proj)
    attention_module.v_proj.kept_out_nodes=kept_indexes
    trim_weight_bias(attention_module.v_proj)
    attention_module.out_proj.kept_in_nodes=kept_indexes
    trim_weight_bias(attention_module.out_proj)
    attention_module.num_heads = remaining_heads
    
    return attention_module

def _trim_feedforward(feedforward_module: nn.Module)->nn.Module:
    if not hasattr(feedforward_module, 'mask_module'):
        return feedforward_module
    mask = feedforward_module.mask_module.binary_mask.squeeze()
    if torch.sum(mask).item()==0:
        print('FFN Layer Trimming')
        return _handle_ffn_layer_trimming(feedforward_module)
    kept_indexes = [x[0] for x in torch.argwhere(mask==1).tolist()]
    feedforward_module.intermediate_dense.kept_out_nodes = kept_indexes
    feedforward_module.output_dense.kept_in_nodes = kept_indexes
    trim_weight_bias(feedforward_module.intermediate_dense)
    trim_weight_bias(feedforward_module.output_dense)
    return feedforward_module

def _trim_conv(conv_module: nn.Module)->nn.Module:
    trim_weight_bias(conv_module.conv)
    return conv_module

def trim_wav2vec(wav2vec: nn.Module)->nn.Module:
    _name = 'network.0'
    conv_layers = []
    for conv_block in range(1, 6):
        cur_block = _name+f'.feature_extractor.conv_layers.{conv_block}'
        cur_layer = _name+f'.feature_extractor.conv_layers.{conv_block}.conv'
        next_layer = _name+f'.feature_extractor.conv_layers.{conv_block+1}.conv'        
        mask = sequential_getattr(wav2vec, cur_block+'.mask_module.binary_mask').squeeze()
        kept_indexes = [x[0] for x in torch.argwhere(mask==1).tolist()]
        sequential_setattr(wav2vec, cur_layer+'.kept_out_nodes', kept_indexes)
        sequential_setattr(wav2vec, next_layer+'.kept_in_nodes', kept_indexes)
        conv_layers.append(cur_layer)
    for n, m in wav2vec.named_modules():
        if hasattr(m, 'mask_module'):
            if n==_name+f'.feature_extractor.conv_layers.{0}':
                continue
            if n==_name+f'.feature_extractor.conv_layers.{6}':
                trimmed_m = _trim_conv(m)
                sequential_setattr(wav2vec, n, trimmed_m)
                continue
            if n+'.conv' in conv_layers:
                trimmed_m = _trim_conv(m)
            elif 'attention' in n:
                trimmed_m = _trim_attention(m)
            elif 'feed_forward' in n:
                trimmed_m = _trim_feedforward(m)
            else:
                raise ValueError(n)
            sequential_setattr(wav2vec, n, trimmed_m)
            sequential_delattr(wav2vec, n+'.mask_module')
    remove_all_hooks(wav2vec, 
                      denylist=[_name+f'.feature_extractor.conv_layers.{0}', 
                                _name+f'.feature_extractor.conv_layers.{6}'])
    return wav2vec

def update_wav2vec_forward(wav2vec: nn.Module)->nn.Module:
    for n, m in wav2vec.named_modules():
        if n.endswith('attention'):
            old_w_shape = m.q_proj.out_features
            new_w_shape = m.q_proj.weight.shape[0]
            if new_w_shape==0:
                m = _handle_attn_layer_trimming(m)
                sequential_setattr(wav2vec, n, m)            
            else:
                old_n_heads = m.num_heads
                head_dim = old_w_shape//old_n_heads
                new_num_heads = new_w_shape//head_dim
                m.num_heads = new_num_heads
                sequential_setattr(wav2vec, n, m)  
        elif n.endswith('feed_forward'):
            if m.intermediate_dense.weight.shape[0]==0:
                _ = _handle_ffn_layer_trimming(m)
                sequential_setattr(wav2vec, n, m) 
    return wav2vec 
