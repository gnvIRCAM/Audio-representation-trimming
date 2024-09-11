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
import yaml

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from audio_example import AudioExample

torch.set_grad_enabled(False)
_EXT = [k.lower() for k in  sf.available_formats().keys()]

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path',
                     None,
                     help='Path to a directory containing audio files',
                     required=True)
flags.DEFINE_string('dataset_name', 
                    None, 
                    required=True, 
                    help="Name of the dataset to be processed \
                        (from \
                        nsynth, gtzan, mtt, \
                        esc50, us8k, fsd, \
                        libri, fluent, commands\
                        )")
flags.DEFINE_string('output_path',
                    ".",
                    help='Output directory for the dataset',
                    required=False)
flags.DEFINE_integer('num_signal',
                     262144,
                     help='Number of audio samples to use during training')
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
    return np.floor(x * (2**15 - 1)).astype(np.int16).tobytes()


def load_audio_chunk(audio_file: tuple, n_signal: int,
                     sr: int) -> Iterable[np.ndarray]:

    path, metadata = audio_file
    # process = subprocess.Popen(
    #     [
    #         'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path, '-ac',
    #         '1', '-af',
    #         'dynaudnorm, silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-30dB',
    #         '-ar',
    #         str(sr), '-f', 's16le', '-'
    #     ],
    #     stdout=subprocess.PIPE,
    # )

    process = subprocess.Popen(
        [
            'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path, '-ac',
            '1', '-af',
            'dynaudnorm, silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-30dB',
            '-f', 's16le', '-'
        ],
        stdout=subprocess.PIPE,
    )

    chunk = process.stdout.read(2 * n_signal)
    i = 0
    while len(chunk) == 2 * n_signal:
        metadata["chunk_number"] = i
        i += 1
        yield chunk, metadata
        chunk = process.stdout.read(2 * n_signal)

    process.stdout.close()

def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm
    
def process_audio_array(audio: Tuple[int, Tuple[bytes,
                                                dict]], env: lmdb.Environment,
                        n_signal: int) -> int:
    
    audio_id, data = audio
    audio_id = audio_id
    audio_samples, metadata = data
    ae = AudioExample()
    ae.put_buffer("waveform", audio_samples, [n_signal])
    ae.put_metadata(metadata)
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            bytes(ae),
        )
    return audio_id


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


def search_for_audios(path_list: Sequence[str], extensions: Sequence[str]):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f'*.{ext}'))
    audios = flatten(audios)
    return audios


def main(dummy):
    FLAGS(sys.argv)

    chunk_load = functools.partial(load_audio_chunk,
                         n_signal=FLAGS.num_signal,
                         sr=FLAGS.sampling_rate)

   # create database
    env = lmdb.open(FLAGS.output_path,
                   map_size=FLAGS.max_db_size * 1024**3,
                   map_async=True,
                   writemap=True,
                   readahead=False)

    print("number of cores: ", FLAGS.num_cores)
    pool = multiprocessing.Pool(FLAGS.num_cores)

    audios = search_for_audios(FLAGS.input_path, _EXT)
    audios = map(str, audios)
    audios = map(os.path.abspath, audios)
    audios = [*audios]
    metadata = [{"path": audio} for audio in audios]

    audios = list(zip(audios, metadata))
    
    # load chunks
    chunks = flatmap(pool, chunk_load, audios)
    chunks = enumerate(chunks)
    
    print("reading chunks")
    
    processed_samples = map(
        functools.partial(process_audio_array,
                env=env,
                n_signal=FLAGS.num_signal), chunks)
    
    pbar = tqdm(processed_samples)
    print("processing samples")
    
    
    for audio_id in pbar:
        n_seconds = FLAGS.num_signal / FLAGS.sampling_rate * audio_id

        pbar.set_description(
            f'dataset length: {timedelta(seconds=n_seconds)}')

    pool.close()
    env.close()


if __name__ == '__main__':
    app.run(main)