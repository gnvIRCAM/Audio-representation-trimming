from absl import app, flags
from dotenv import load_dotenv
load_dotenv('.env')
import os
from shutil import copyfile
import sys

import torch 
import gin 
gin.add_config_file_search_path('configs/')

from datasets import *
from metrics import *
from networks import *
from training import *
from trim import *

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('config', default=None, required=False, 
                          help='Config files to run experiments')
flags.DEFINE_integer('device', default=-1, required=False, 
                     help='Cuda device to train on (-1: cpu)')
flags.DEFINE_string('run_name', default=None, required=True, 
                    help='Where to store checkpoints+logs')
flags.DEFINE_string('db', default=None, required=True, 
                    help='Path to lmdb dataset')
flags.DEFINE_integer('bs', default=8, required=False, help='Batch size')
flags.DEFINE_integer('fold', default=0, required=False, help='Data fold to use for training/validation')
flags.DEFINE_string('resume', default=None, required=False, 
                    help='Resume experiment from corresponding checkpoint')
flags.DEFINE_string('pretrained_mask', default=None, required=False, 
                    help="Path to checkpoint to use for pre-trimming")

def main(argv):
    assert FLAGS.config is not None or FLAGS.resume is not None, 'Either a config file or a run dir. should be provided'
    if FLAGS.device==-1:
        device='cpu'
    else:
        device=f'cuda:{FLAGS.device}'
    train_loader, val_loader, _, dataset_sr, num_classes = make_loaders(FLAGS.db, FLAGS.bs, num_workers=8, 
                                                     fold=FLAGS.fold)
    gin.parse_config_files_and_bindings(FLAGS.config, bindings=[
        f'BS = {FLAGS.bs}', 
        f'DATASET_SR = {dataset_sr}', 
        f'NUM_CLASSES = {num_classes}'
    ])
    os.makedirs(FLAGS.run_name, exist_ok=True)
    model = combine_fm_and_head()
    print(f'Number of parameters : {round(sum([p.numel() for p in model.parameters()])/1e6, 2)}M')
    model = make_masks(model)
    if FLAGS.pretrained_mask is not None:
        copyfile(FLAGS.pretrained_mask, os.path.join(FLAGS.run_path, 'source.pt'))
        weights = torch.load(FLAGS.pretrained_mask, map_location='cpu')['model']
        model.load_state_dict(weights)
        trim_model(model)
        del weights
    trainer = Trainer(run_name=FLAGS.run_name, device=device)
    with open(os.path.join(FLAGS.run_name, 'data.txt'), 'w') as f:
        f.write(
            f'Dataset : {FLAGS.db}\nFold: {FLAGS.fold}'
        )
    trainer.fit(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader
    )


if __name__=='__main__':
    app.run(main)