import typing as tp

from calflops import calculate_flops
import torch.nn as nn 

def get_flops_macs(model: nn.Module, 
              input_dur: float=4)->tp.Tuple[str]:
    sr = model.resampler.new_freq
    input_shape = (1, int(sr*input_dur))
    flops, macs, _ = calculate_flops(model, 
                                     input_shape, 
                                     output_as_string=True, 
                                     output_precision=4, 
                                     print_detailed=False, 
                                     print_results=False
                                     )
    return flops, macs

def compute_model_memory(model: nn.Module) -> int:
    """
    Returns size (in bits) of the parameters of a model
    """
    return sum([p.numel() for p in model.parameters()])*32

def convert_bits(bits: int, unit: str = 'Go') -> float:
    """
    Convert size in bits into target unit (from Go, Mo)
    """
    assert unit in ['Mo', 'Go'], NotImplementedError()
    if unit=='Go':
        return bits*1.25*1e-10
    elif unit=='Mo':
        return bits*1.25*1e-7

def get_num_params(model: nn.Module, layer_idx: int = -1) -> int:
    if layer_idx==-1:
        num_params = sum(list([p.numel() for p in model.parameters()]))
    else:
        num_params = 0
        for n, p in model.named_parameters():
            if 'layers' in n:
                cur_layer = int(n.split('layers')[-1].split('.')[1])
                if cur_layer >= layer_idx:
                    break
                num_params+=p.numel()
    return num_params
            