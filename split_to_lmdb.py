from absl import app, flags
from datetime import timedelta
from itertools import repeat
import functools
import multiprocessing
import os
import pathlib
import soundfile as sf
import subprocess
import sys
from typing import Callable, Iterable, Sequence, Tuple

import librosa as li
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from datasets import *

torch.set_grad_enabled(False)
_EXT = [k.lower() for k in  sf.available_formats().keys()]

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path',
                     None,
                     help='Path to a directory containing audio files',
                     required=True)
flags.DEFINE_string('output_path',
                    ".",
                    help='Output directory for the dataset',
                    required=False)
flags.DEFINE_string('dataset_name', 
                    None, 
                    required=True, 
                    help="Name of the dataset to be processed \
                        (from \
                        nsynth, gtzan, mtt, \
                        esc50, us8k, fsd, \
                        libri, fluent, commands\
                        )")
flags.DEFINE_integer('num_cores',
                     8,
                     help='Number of cores for multiprocessing')
flags.DEFINE_integer('max_db_size',
                     180,
                     help='Maximum size (in GB) of the dataset')
flags.DEFINE_bool('dyndb',
                  default=True,
                  help="Allow the database to grow dynamically")


def float_array_to_int16_bytes(x):
    return np.floor(x * (2**15 - 1)).astype(np.int16)

def load_audio(audio_file: tuple) -> Iterable[np.ndarray]:
    path, metadata = audio_file
    y, _ = li.load(path)
    y = y.squeeze()
    y = float_array_to_int16_bytes(y)
    yield y, metadata

def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm

def process_audio_array(audio: Tuple[int, Tuple[bytes,dict]], 
                        env: lmdb.Environment) -> int:
    
    audio_id, data = audio
    audio_samples, metadata = data
    duration = audio_samples.shape[0]
    ae = AudioExample()
    ae.put_array("waveform", audio_samples, dtype=np.int16)
    ae.put_metadata(metadata)
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            bytes(ae),
        )
    return duration

def flatmap(pool: multiprocessing.Pool,
            func: Callable,
            iterable: Iterable,
            chunksize=None):
    queue = multiprocessing.Manager().Queue(maxsize=os.cpu_count())
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()

def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)

def search_for_audios(path_list: Sequence[str] | str, extensions: Sequence[str]):
    if isinstance(path_list, str):
        path_list = [path_list]
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        print(p)
        for ext in extensions:
            audios.append(p.rglob(f'*.{ext}'))
    audios = flatten(audios)
    return audios

def main(argv):
    FLAGS(sys.argv)

    labels = load_metadata(f'datasets/data/metadata/{FLAGS.dataset_name}_metadata.json')
    labels = {os.path.join(FLAGS.input_path, k): v for k, v in labels.items()}

   # create database
    env = lmdb.open(FLAGS.output_path,
                   map_size=FLAGS.max_db_size * 1024**3,
                   map_async=True,
                   writemap=True,
                   readahead=False)

    print("Number of cores: ", FLAGS.num_cores)
    pool = multiprocessing.Pool(FLAGS.num_cores)

    print('Searching for audios')
    audios = search_for_audios(FLAGS.input_path, _EXT)
    audios = map(str, audios)
    audios = list(map(os.path.abspath, audios))
    print(f'Found  {len(audios)} audio files')
    _, _SR = sf.read(audios[0])
    metadata = [{"path": audio, "metadata": labels[audio]} for audio in audios]
    audios = list(zip(audios, metadata))
    
    chunks = flatmap(pool, load_audio, audios)
    chunks = enumerate(chunks)
    
    print("Reading audios")
    
    processed_samples = map(
        functools.partial(process_audio_array,
                env=env), chunks)
    
    pbar = tqdm(processed_samples)
    print("Processing samples")
    
    n_seconds = 0
    for audio_length in pbar:
        n_seconds +=  audio_length/ _SR

        pbar.set_description(
            f'dataset length: {timedelta(seconds=n_seconds)}')

    pool.close()
    env.close()

if __name__ == '__main__':
    app.run(main)