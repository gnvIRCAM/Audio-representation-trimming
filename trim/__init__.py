from .trim_clap import *
from .trim_musicfm import *
from .trim_wav2vec import *
from .utils import *

import gin

@gin.configurable(module='trim', denylist=['model'])
def trim_model(model, trimmer):
    return trimmer(model)