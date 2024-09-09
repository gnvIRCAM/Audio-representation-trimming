import typing as tp 
from types import MethodType

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .utils import *

def _trim_input_group_conv(module, input):
    return torch.index_select(input[0], dim=1, 
                              index=torch.tensor(module.kept_out_nodes).to(input[0].device))
    
def _trimmed_rotary_embedding(self, hidden_states, relative_position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, sequence_length, self.old_num_heads, self.head_size)

        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.old_num_heads * self.head_size)

        return hidden_states

def _musicfm_mha_layertrim_forward(self,
                                   hidden_states: torch.Tensor,
                                   attention_mask = None,
                                   relative_position_embeddings = None,
                                   output_attentions: bool = False,
                                ):
    # self-attention mechanism
    batch_size, sequence_length, hidden_size = hidden_states.size()
    bias = self.out_bias
    if bias is None:
        return torch.zeros_like(hidden_states), None
    bias: torch.Tensor = bias.data.unsqueeze(0).unsqueeze(0)
    output = bias.repeat(batch_size, sequence_length, 1) # B, L, E
    return output, None

def _handle_layer_trimming(attention_module: nn.Module)-> nn.Module:
    attention_module.out_bias = attention_module.linear_out.bias
    delattr(attention_module, 'linear_q')
    delattr(attention_module, 'linear_k')
    delattr(attention_module, 'linear_v')
    delattr(attention_module, 'linear_out')
    attention_module.forward = MethodType(_musicfm_mha_layertrim_forward, attention_module)
    return attention_module

def _trim_attention(attention_module: nn.Module
                   )->nn.Module:
    if not hasattr(attention_module, 'mask_module'):
        return attention_module
    mask = attention_module.mask_module.binary_mask.squeeze()
    if torch.sum(mask).item()==0:
        print('Layer Trimming')
        return _handle_layer_trimming(attention_module)
    kept_heads = [x[0] for x in torch.argwhere(mask==1).tolist()]
    num_heads = len(mask)
    remaining_heads = torch.count_nonzero(mask).item()
    embed_dim = attention_module.linear_q.weight.shape[0]
    head_size = embed_dim//num_heads
    attention_module.old_num_heads = attention_module.num_heads
    def compute_head_indexes(head: int):
        return [(head*head_size)+idx for idx in range(head_size)]
    kept_indexes = []
    
    for m in kept_heads:
        kept_indexes+=compute_head_indexes(m)
    attention_module.kept_indexes = kept_indexes
        
    attention_module.linear_q.kept_out_nodes=kept_indexes
    trim_weight_bias(attention_module.linear_q)
    attention_module.linear_k.kept_out_nodes=kept_indexes
    trim_weight_bias(attention_module.linear_k)
    attention_module.linear_v.kept_out_nodes=kept_indexes
    trim_weight_bias(attention_module.linear_v)
    attention_module.linear_out.kept_in_nodes=kept_indexes
    trim_weight_bias(attention_module.linear_out)
    attention_module.num_heads = remaining_heads
    attention_module._apply_rotary_embedding = MethodType(_trimmed_rotary_embedding, attention_module)
    return attention_module

def _trim_conformer(conformer_module: nn.Module)->nn.Module:
    if not hasattr(conformer_module, 'mask_module'):
        return conformer_module
    mask = conformer_module.mask_module.binary_mask.squeeze()
    kept_channels = [x[0] for x in torch.argwhere(mask==1).tolist()]   
    conformer_module.depthwise_conv.kept_out_nodes = kept_channels
    if conformer_module.depthwise_conv.groups!=1:
        conformer_module.depthwise_conv.groups=len(kept_channels)
        conformer_module.depthwise_conv.register_forward_pre_hook(_trim_input_group_conv)
    conformer_module.batch_norm.kept_in_nodes = kept_channels
    conformer_module.pointwise_conv2.kept_in_nodes = kept_channels
    trim_weight_bias(conformer_module.depthwise_conv)
    trim_weight_bias(conformer_module.batch_norm)
    trim_weight_bias(conformer_module.pointwise_conv2)
    return conformer_module

def _trim_feedforward(feedforward_module: nn.Module)->nn.Module:
    if not hasattr(feedforward_module, 'mask_module'):
        return feedforward_module
    mask = feedforward_module.mask_module.binary_mask.squeeze()
    kept_indexes = [x[0] for x in torch.argwhere(mask==1).tolist()]
    feedforward_module.intermediate_dense.kept_out_nodes = kept_indexes
    feedforward_module.output_dense.kept_in_nodes = kept_indexes
    trim_weight_bias(feedforward_module.intermediate_dense)
    trim_weight_bias(feedforward_module.output_dense)
    return feedforward_module

def trim_musicfm(musicfm: nn.Module, layer_stop_idx: int=-1):
    _name = 'network.0.conformer.layers'
    trim_layers = []
    for n, m in musicfm.named_modules():
        if _name not in n:
            continue
        if n!=_name:
            layer_idx = n.split(_name)[1].split('.')[1]
            if layer_stop_idx!=-1 and int(layer_idx)>=layer_stop_idx:
                continue
        if hasattr(m, 'mask_module'):
            if 'ffn' in n:
                trimmed_m = _trim_feedforward(m)
            elif 'self_attn' in n:
                trimmed_m = _trim_attention(m)
            elif 'conv_module' in n:
                trimmed_m = _trim_conformer(m)
            sequential_setattr(musicfm, n, trimmed_m)
            trim_layers.append(n)
    for t_n in trim_layers:
        sequential_delattr(musicfm, t_n+'.mask_module')
    remove_all_hooks(musicfm)
    return musicfm

def update_musicfm_forward(musicfm: nn.Module, 
                           trimmed_layers: tp.List[str]=[])->nn.Module:
    for n, m in musicfm.named_modules():
        if n.endswith('self_attn'):
            old_w_shape = m.linear_q.out_features
            new_w_shape = m.linear_q.weight.shape[0]
            old_num_heads = m.num_heads
            head_dim = old_w_shape//old_num_heads
            new_num_heads = new_w_shape//head_dim
            if new_num_heads==0 or n in trimmed_layers:
                m = _handle_layer_trimming(m)
            else:
                m.num_heads = new_num_heads
