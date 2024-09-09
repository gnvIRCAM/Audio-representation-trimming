from collections import OrderedDict
import typing as tp

import torch.nn as nn

def sequential_hasattr(obj: nn.Module,
                       name: str
                       ) -> bool:
    """
    hasattr() handler for nn.Sequential()
    """
    name, res = name.split('.')[0], name.split('.')[1:]
    if not len(res):
        if not name.isdigit():
            return hasattr(obj, name)
        else:
            return len(obj) > int(name)
    else:
        res = '.'.join(res)
        if not name.isdigit() and hasattr(obj, name):
            return sequential_hasattr(getattr(obj, name), res)
        elif len(obj) > int(name):
            return sequential_hasattr(obj[int(name)], res)
        else:
            return False

def sequential_delattr(obj: nn.Module,
                       name: str
                       ) -> None:
    """
    delattr() handler for nn.Sequential()
    """
    name, res = name.split('.')[0], name.split('.')[1:]
    if not len(res):
        delattr(obj, name)
    else:
        res = '.'.join(res)
        if not name.isdigit():
            return sequential_delattr(getattr(obj, name), res)
        else:
            return sequential_delattr(obj[int(name)], res)

def sequential_getattr(obj: nn.Module,
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

def sequential_setattr(obj: nn.Module,
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

def has_no_children(module: nn.Module
                    ) -> bool:
    return not len([c for c in module.children()])

def has_weight(module: nn.Module,
               name: tp.Optional[str] = None,
               ) -> bool:
    """
    Given an nn.Module/name of a layer, will return True
    if it has trainable parameters
    """
    if hasattr(module, 'weight'):
        return True
    if name is None:
        if not has_no_children(module):
            return False
        return bool(len(list(module.named_parameters())))
    else:
        module = sequential_getattr(module, name)
        if not has_no_children(module):
            return False
        return bool(len(list(module.named_parameters())))

def has_in_nodes_first(module: nn.Module,
                       name: tp.Optional[str] = None) -> bool:
    """
    For a weighted layer (FC, Conv2d...), returns True if the weights matrix 
    is not tranposed before being applied
    """
    m = module
    if name is not None:
        m = sequential_getattr(module, name)
    for in_nodes_layers in [nn.ConvTranspose1d, nn.ConvTranspose2d,
                            nn.ConvTranspose3d, nn.GRU, nn.RNN]:
        if isinstance(m, in_nodes_layers):
            return True
    return False

def is_norm_layer(module: nn.Module,
                  name: tp.Optional[str] = None
                  ) -> bool:
    """
    Given an nn.Module/name of a layer, will return True
    if it corresponds to a normalization layer
    """
    m = module
    if name is not None:
        m = sequential_getattr(module, name)
    return 'norm' in str(m.__class__)

def trim_weight_bias(module: nn.Module
                     ) -> None:
    """
    Trims parameters and buffers of a nn.Module
    """
    if not has_weight(module):
        return

    if hasattr(module, 'kept_in_nodes'):
        kept_in_nodes = module.kept_in_nodes
    else:
        kept_in_nodes = None
    if hasattr(module, 'kept_out_nodes'):
        kept_out_nodes = module.kept_out_nodes
    else:
        kept_out_nodes = None

    w = module.weight.data

    if kept_in_nodes is not None and w is not None:
        if has_in_nodes_first(module):
            w = w[kept_in_nodes]
        elif is_norm_layer(module):
            w = w[kept_in_nodes]
        else:
            w = w[:, kept_in_nodes]
    if kept_out_nodes is not None and w is not None:
        if has_in_nodes_first(module):
            w = w[:, kept_out_nodes]
        else:
            w = w[kept_out_nodes]

    module.weight = nn.Parameter(w)

    # Handle bias
    if hasattr(module, 'bias') and module.bias is not None:
        b = module.bias
        if kept_out_nodes is not None:
            b = b[kept_out_nodes]
        elif kept_in_nodes is not None and 'batchnorm' in str(module.__class__):
            b = b[kept_in_nodes]
        module.bias = nn.Parameter(b)

    # Handle running_mean and running_var
    if hasattr(module, 'running_mean') and module.running_mean is not None:
        running_mean = module.running_mean.data
        running_var = module.running_var.data
        if kept_in_nodes is not None:
            running_mean = running_mean[kept_in_nodes]
            running_var = running_var[kept_in_nodes]
        module.running_mean = running_mean
        module.running_var = running_var
    return

def remove_all_hooks(model: nn.Module, 
                     denylist: tp.List[str]=[])->None:
    """
    Remove all registered forward hooks for a given model
    """
    for n, m in model.named_modules():
        if n in denylist:
            continue
        if hasattr(m, '_forward_hooks'):
            m._forward_hooks = OrderedDict()