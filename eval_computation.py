from absl import app, flags
from dotenv import load_dotenv
load_dotenv('.env')
import os
import time 

import torch 
import gin 
gin.add_config_file_search_path('configs/')

from datasets import *
from metrics import *
from networks import *
from training import *
from trim import *

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS
flags.DEFINE_string('folder', default=None, required=False, 
                          help='Folder of checkpoints to evaluate')
flags.DEFINE_string('ckpt', default=None, required=False, 
                          help='Checkpoint to evaluate')
flags.DEFINE_integer('device', default=-1, required=False, 
                     help='Cuda device to train on (-1: cpu)')


def main(argv):
    assert FLAGS.folder is not None or FLAGS.ckpt is not None, 'Either a folder of checkpoints or a checkpoint should be specified'
    if FLAGS.folder is not None:
        checkpoints = [os.path.join(FLAGS.folder, ckpt) for ckpt in os.listdir(FLAGS.folder) if ckpt.endswith('.pt')]
        checkpoints = sorted(checkpoints)
    else:
        checkpoints = [FLAGS.ckpt]
    run_path = os.path.join(*checkpoints[0].split('/')[:-1])
    config = os.path.join(run_path, 'config.gin')
    if FLAGS.device==-1:
        device='cpu'
    else:
        device=f'cuda:{FLAGS.device}'
        
    with open(os.path.join(run_path, 'data.txt'), 'r') as f:
        dataset_metadata = f.read().split('\n')
        db = dataset_metadata[0].replace('Dataset : ', '')
        fold = int(dataset_metadata[1].replace('Fold: ', ''))
    bs = 8 # TO CHANGE
    _, _, test_loader, _, _ = make_loaders(db, bs, num_workers=8, 
                                           fold=fold)
    
    gin.parse_config_file(config)
    data_sr = gin.query_parameter('%DATA_SR')

    # Create a 4-seconds long sample for speed test
    x = torch.randn((1, 4*data_sr), device=device) 

    computation_metadata = {}
    
    # Untrimmed model
    model = combine_fm_and_head().to(device)
    model.eval()
    print('Evaluating untrimmed model', end="\r")
    flops, macs = get_flops_macs(model, input_dur=4)
    model_memory = compute_model_memory(model)
    model_memory = convert_bits(model_memory, unit='Mo')
    try:
        layer_idx = gin.query_parameter('%LAYER_IDX')
    except:
        layer_idx=-1
    num_params = get_num_params(model, layer_idx=layer_idx)
    computation_metadata['Full model'] = {
        'flops': flops, 
        'macs': macs, 
        'memory': model_memory, 
        'num_params': num_params
    } 
    t_start = time.time()
    for _ in range(1000):
        _ = model(x)
    t_end = time.time()
    computation_metadata['Full model']['Inference time'] = (t_end-t_start)/1000
    
    # Trimmed model(s)
    for idx, ckpt in enumerate(checkpoints):
        print(f'Evaluating checkpoint {ckpt} ({idx+1}/{len(checkpoints)})', end="\r")
        model = combine_fm_and_head()
        model = make_masks(model)
        weights = torch.load(ckpt, map_location='cpu')['model']
        model.load_state_dict(weights)

        if isinstance(model, End2EndCLAP):
            trimmer = trim_clap
        if isinstance(model, End2EndMusicFM):
            trimmer = trim_musicfm
        if isinstance(model, End2EndWav2Vec):    
            trimmer = trim_wav2vec
            
        trim_model(model, trimmer=trimmer)
        model = model.to(device)
        flops, macs = get_flops_macs(model, input_dur=4)
        model_memory = compute_model_memory(model)
        model_memory = convert_bits(model_memory, unit='Mo')
        num_params = get_num_params(model, layer_idx=layer_idx)
        computation_metadata[ckpt] = {
            'flops': flops, 
            'macs': macs, 
            'memory': model_memory, 
            'num_params': num_params
        } 
        t_start = time.time()
        for _ in range(1000):
            _ = model(x)
        t_end = time.time()
        computation_metadata[ckpt]['Inference time'] = (t_end-t_start)/1000
    with open(os.path.join(run_path, 'computation.json'), 'w') as f:
        json.dump(computation_metadata, f, indent=2)
        
if __name__=='__main__':
    app.run(main)
        