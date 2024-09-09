import math 
from types import MethodType
import typing as tp

import gin 
import torch
import torch.nn as nn 

from .mask_module import MaskModule

__all__ = ['make_clap_masks', 'make_musicfm_masks', 'make_wav2vec_masks']

def sequential_getattr(obj: tp.Any,
                       name: str
                       ) -> bool:
    """
    getattr() handler for nn.Sequential()
    """
    name, res = name.split('.')[0], name.split('.')[1:]
    if not len(res):
        if not name.isdigit():
            return getattr(obj, name)
        else:
            return obj[int(name)]
    else:
        res = '.'.join(res)
        if not name.isdigit():
            return sequential_getattr(getattr(obj, name), res)
        else:
            return sequential_getattr(obj[int(name)], res)

def sequential_setattr(obj: tp.Any,
                       name: str,
                       val: tp.Any
                       ) -> None:
    """
    setattr() handler for nn.Sequential()
    """
    name, res = name.split('.')[0], name.split('.')[1:]
    if not len(res):
        setattr(obj, name, val)
    else:
        res = '.'.join(res)
        if not name.isdigit():
            return sequential_setattr(getattr(obj, name), res, val)
        else:
            return sequential_setattr(obj[int(name)], res, val)

@gin.configurable(module='models', denylist=['model'])
def make_clap_masks(model: nn.Module, 
                    layers: tp.Dict[str, tp.Dict[str, int]],
                    mask_module: nn.Module = MaskModule
                    ) -> nn.Module:
    """
    Model : model to apply masks on
    Layers : Layers to mask outputs. For each layer, the number of features, 
    as well as the index of the feature (in the output) must be specified
    """
    for n, m in model.named_modules():
        if n in layers.keys():
            num_features = layers[n]['num_features']
            feature_dim = layers[n].get('feature_dim', 1)
            m.mask_module = mask_module(num_features, feature_dim)
            def hook(model, input, output):
                return model.mask_module(output)
            m.register_forward_hook(hook)
    return model

def _musicfm_masked_ffn_forward(self, hidden_states):
    hidden_states = self.intermediate_dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    hidden_states = self.intermediate_dropout(hidden_states)

    if hasattr(self, 'mask_module'):
        hidden_states = self.mask_module(hidden_states)

    hidden_states = self.output_dense(hidden_states)
    hidden_states = self.output_dropout(hidden_states)
    return hidden_states

def _musicfm_masked_conformer_forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        
        if hasattr(self, 'mask_module'):
            hidden_states = self.mask_module(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

def _musicfm_masked_self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None,
    relative_position_embeddings = None,
    output_attentions: bool = False,
):
    # self-attention mechanism
    batch_size, sequence_length, hidden_size = hidden_states.size()

    # make sure query/key states can be != value states
    query_key_states = hidden_states
    value_states = hidden_states

    if self.position_embeddings_type == "rotary":
        if relative_position_embeddings is None:
            raise ValueError(
                "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
            )
        query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

    # project query_key_states and value_states
    query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
    key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
    value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

    # => (batch, head, time1, d_k)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if self.position_embeddings_type == "relative":
        if relative_position_embeddings is None:
            raise ValueError(
                "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                " 'relative'"
            )
        # apply relative_position_embeddings to qk scores
        # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
        scores = self._apply_relative_embeddings(
            query=query, key=key, relative_position_embeddings=relative_position_embeddings
        )
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

    # apply attention_mask if necessary
    if attention_mask is not None:
        scores = scores + attention_mask

    # => (batch, head, time1, time2)
    probs = torch.softmax(scores, dim=-1)
    probs = self.dropout(probs)

    if hasattr(self, 'mask_module'):
        probs = self.mask_module(probs)

    # => (batch, head, time1, d_k)
    hidden_states = torch.matmul(probs, value)

    # => (batch, time1, hidden_size)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
    hidden_states = self.linear_out(hidden_states)

    return hidden_states, probs

@gin.configurable(module='models', denylist=['model'])
def make_musicfm_masks(model: nn.Module, 
                       mask_module: nn.Module = MaskModule,  
                       layer_stop_idx: int=-1, 
                       exclude_layers: tp.List[str]=[]
                       )->nn.Module:
    trim_layers = {}
    for n, m in model.named_modules():
        if n in exclude_layers:
            continue
        if 'layers' in n:
            name_split = n.split('layers')[-1].split('.')
            if len(name_split)==1:
                layer_idx=0
            else:
                layer_idx = int(name_split[1])
            if (layer_stop_idx != -1) and layer_idx>=layer_stop_idx:
                break
            if hasattr(m, 'num_heads'):
                trim_layers[n]={'num_features':m.num_heads}
            end = n.split('.')[-1]
            if 'ffn' in end and 'layer_norm' not in end:
                trim_layers[n]={'num_features':m.intermediate_dense.out_features, 'feature_dim':-1}
            if end=='conv_module':
                trim_layers[n]={'num_features':m.depthwise_conv.weight.shape[0]}
    for layer, params in trim_layers.items():
        sequential_setattr(model, layer+'.mask_module', 
                                   mask_module(**params))
        masked_module = sequential_getattr(model, layer)
        if 'ffn' in layer:
            masked_module.forward = MethodType(_musicfm_masked_ffn_forward, masked_module)
        elif 'self_attn' in layer:
            masked_module.forward = MethodType(_musicfm_masked_self_attn_forward, masked_module)
        else:
            masked_module.forward = MethodType(_musicfm_masked_conformer_forward, masked_module)
    return model
                
def _wav2vec_masked_conv_forward(self, x, length,):
    """
    Args:
        x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
        length (Tensor or None, optional): Shape ``[batch, ]``.
    Returns:
        Tensor: Shape ``[batch, out_channels, out_frames]``.
        Optional[Tensor]: Shape ``[batch, ]``.
    """
    x = self.conv(x)
    if self.layer_norm is not None:
        x = self.layer_norm(x)
    x = nn.functional.gelu(x)

    if hasattr(self, 'mask_module'):
        x = self.mask_module(x)
    if length is not None:
        length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
        # When input length is 0, the resulting length can be negative. So fix it here.
        length = torch.max(torch.zeros_like(length), length)
    return x, length

def _wav2vec_masked_mha_forward(self,
                                x,
                                attention_mask = None,
                                position_bias = None,key_padding_mask = None,
                                ):
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False
        )
        if hasattr(self, 'mask_module'):
            attn_output = self.mask_module(attn_output)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        return output, None  # Necessary for compatibility with WavLMSelAttention

def _wav2vec_masked_ffn_forward(self, x):
    """
    Args:
        x (Tensor): shape: `(batch, sequence_length, io_features)`
    Returns:
        x (Tensor): shape: `(batch, sequence_length, io_features)`
    """
    x = self.intermediate_dense(x)
    x = torch.nn.functional.gelu(x)
    x = self.intermediate_dropout(x)
    if hasattr(self, 'mask_module'):
        x=self.mask_module(x)
    
    x = self.output_dense(x)
    x = self.output_dropout(x)
    return x

@gin.configurable(module='models', denylist=['model'])
def make_wav2vec_masks(model: nn.Module, 
                       mask_module: nn.Module = MaskModule,  
                       exclude_layers: tp.List[str]=[], 
                       layer_stop_idx: int=-1,
                       allowlist: tp.Optional[tp.List[str]]=None, 
                       )->nn.Module:
    layer_idx=0
    for n, m in model.named_modules():
        if n in exclude_layers:
            continue
        if allowlist is not None and (n not in allowlist):
            continue
        if 'conv_layers' in n:
            if n.split('.')[-2]=='conv_layers':
                m.mask_module = mask_module(m.conv.weight.shape[0])
                m.forward = MethodType(_wav2vec_masked_conv_forward, m)
        if n.endswith('feed_forward'):
            m.mask_module = mask_module(m.intermediate_dense.weight.shape[0], feature_dim=-1)
            m.forward = MethodType(_wav2vec_masked_ffn_forward, m)
        if n.endswith('attention'):
            m.mask_module = mask_module(m.num_heads)
            m.forward = MethodType(_wav2vec_masked_mha_forward, m)  
            layer_idx+=1
            if (layer_stop_idx!=-1) and (layer_idx>=layer_stop_idx):
                break
    return model