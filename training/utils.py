import math
import os
import typing as tp
import sys
sys.path.append('..')

import gin
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
import torch.utils
import torch.utils.data
from transformers import get_constant_schedule_with_warmup, get_constant_schedule

from networks import End2EndCLAP, End2EndMusicFM, End2EndWav2Vec, MaskModule

def gather_masks(masked_model: nn.Module
                 ) -> tp.Dict[str, nn.Parameter]:
    masks = {}
    for n, m in masked_model.named_modules():
        if hasattr(m, 'mask_module'):
            masks[n] = m.mask_module.mask
    return masks

@gin.configurable(module='train', denylist=['model'])
def make_optimizer(
    model: nn.Module, 
    lr: float, 
    denylist: tp.List[str] = [], 
    exclude_masks: bool = False, 
    include_fm: bool = False
    ) -> Adam:
    params = []
    mask_params = {'name': 'masks', 'params':[]}
    for n, m in model.named_modules():
        if hasattr(m, 'mask_module') and (n not in denylist):
            mask_params['params']+=list(m.mask_module.parameters())
    if len(mask_params) and not exclude_masks:
        params.append(mask_params)
    params.append({'params':list(model.head.parameters()), 
             'name': 'head'})
    if include_fm:
        params.append({'params':list(model.foundation_model.parameters()), 
                'name': 'foundation_model'})
    return Adam(params, lr=lr, betas=[.9, .999])

@gin.configurable(module='train', denylist=['optimizer'])
def warmup_lr_scheduler(optimizer: torch.optim.Optimizer, 
                        num_warmup: int):
    return get_constant_schedule_with_warmup(optimizer, num_warmup)

@gin.configurable(module='train', denylist=['optimizer'])
def constant_lr_scheduler(optimizer: torch.optim.Optimizer):
    return get_constant_schedule(optimizer)

def _init_transformer_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

@gin.configurable(module='train')
def init_wav2vec_weights(model: nn.Module)->nn.Module:
    for n, m in model.foundation_model.named_modules():
        m_type = str(type(m))
        if 'ConvLayerBlock' in m_type:
            nn.init.kaiming_normal_(m.conv.weight)
        elif 'ConvolutionalPositionalEmbedding' in m_type:
            # normalize the weight to normal distribution.
            std = math.sqrt(4.0 / (m.embed_dim * m.kernel_size))
            nn.init.normal_(m.conv.weight, mean=0.0, std=std)
            nn.init.constant_(m.conv.bias, 0.0)
        elif 'SelfAttention' in m_type:
            # normalize the query, key, value, and out_proj parameters in self attention module.
            nn.init.xavier_uniform_(m.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(m.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(m.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(m.out_proj.weight)
            nn.init.constant_(m.out_proj.bias, 0.0)
        elif 'Transformer' in m_type:
            m.apply(_init_transformer_params)
        else:
            pass
    return model

@gin.configurable(module='train')
def init_musicfm_weights(model: nn.Module) -> nn.Module:
    for n, m in model.foundation_model.named_modules():
        """Initialize the weights"""
        # Wav2Vec2ForPreTraining last 2 linear layers need standard Linear init.
        m_type = str(type(m))
        if 'Wav2Vec2ConformerForPreTraining' in m_type:
            m.project_hid.reset_parameters()
            m.project_q.reset_parameters()
            m.project_hid._is_hf_initialized = True
            m.project_q._is_hf_initialized = True
        # gumbel softmax requires special init
        elif 'Wav2Vec2ConformerGumbelVectorQuantizer' in m_type:
            m.weight_proj.weight.data.normal_(mean=0.0, std=1)
            m.weight_proj.bias.data.zero_()
            nn.init.uniform_(m.codevectors)
        elif 'Wav2Vec2ConformerSelfAttention' in m_type:
            if hasattr(m, "pos_bias_u"):
                nn.init.xavier_uniform_(m.pos_bias_u)
            if hasattr(m, "pos_bias_v"):
                nn.init.xavier_uniform_(m.pos_bias_v)
        elif 'Wav2Vec2ConformerPositionalConvEmbedding' in m_type:
            nn.init.normal_(
                m.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (m.conv.kernel_size[0] * m.conv.in_channels)),
            )
            nn.init.constant_(m.conv.bias, 0)
        elif 'Wav2Vec2ConformerFeatureProjection' in m_type:
            k = math.sqrt(1 / m.projection.in_features)
            nn.init.uniform_(m.projection.weight, a=-k, b=k)
            nn.init.uniform_(m.projection.bias, a=-k, b=k)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=.02)

            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)

            if m.bias is not None:
                k = math.sqrt(m.groups / (m.in_channels * m.kernel_size[0]))
                nn.init.uniform_(m.bias, a=-k, b=k)
    return model

@gin.configurable(module='train')
def init_clap_weights(model: nn.Module):
    for n, p in model.foundation_model.named_parameters():
        if 'logit_scale' in n or ('spectrogram' in n) or ('logmel' in n):
            continue
        nn.init.normal_(p)
        if 'bias' in n:
            nn.init.constant_(p, 0.)     
    for n, m in model.foundation_model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.)   
            nn.init.constant_(m.bias, 0.)
    return model   


@gin.configurable(module='train')
def scratch_denylist(model: nn.Module):
    """
    For CLAP, spectrogram extractor is listed as trainable, 
    but we don't want to train it, so we dont include it in the optimizer
    """
    denylist= []
    if not isinstance(model, End2EndCLAP):
        return denylist
    for n, _ in model.foundation_model.named_parameters():
        if 'logit_scale' in n or ('spectrogram' in n) or ('logmel' in n):
            denylist.append(n)
    return denylist
