import typing as tp

from calflops import calculate_flops
import torch 
import torch.nn as nn 

def get_flops_macs(model: nn.Module, 
              input_dur: float=4)->tp.Tuple[str]:
    sr = model.resampler.target_sr
    input_shape = (1, 1, int(sr*input_dur))
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
