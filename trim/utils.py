from collections import OrderedDict
import typing as tp

import torch.nn as nn

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