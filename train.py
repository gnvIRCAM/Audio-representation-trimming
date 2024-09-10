from absl import app, flags
from dotenv import load_dotenv
import os
import sys
load_dotenv('.env')

import torch 
import gin 
gin.add_config_file_search_path('configs/')

from datasets import *
from metrics import *
from networks import *
from trim import *

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('config', default=None, required=True, 
                          help='Config files to run experiments')
flags.DEFINE_integer('device', default=-1, required=False, 
                     help='Cuda device to train on (-1: cpu)')
flags.DEFINE_string('db', default=None, required=True, 
                    help='Path to lmdb dataset')
flags.DEFINE_string('run_name', default=None, required=True, 
                    help='Where to store checkpoints+logs')
flags.DEFINE_string('resume', default=None, required=False, 
                    help='Resume experiment from corresponding checkpoint')
flags.DEFINE_string('pretrained_mask', default=None, required=False, 
                    help="Path to checkpoint to use for pre-trimming")

def main(argv):
    pass

if __name__=='__main__':
    gin.parse_config_files_and_bindings(FLAGS.config, bindings=None)
    app.run(main)