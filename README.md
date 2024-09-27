# Audio representation trimming

Repository for the paper "Keep what you need : extracting efficient subnetworks from large audio representation models", by David Genova, Philippe Esling, and Tom Hurlin.

## Installation

Along a functional pytorch installation, please run ` pip install -r requirements.txt ` to install the required dependencies.
In addition, you will have to install the foundation models from these repositories :

- MusicFM : [https://github.com/minzwon/musicfm](https://github.com/minzwon/musicfm)
- MSCLAP : [https://github.com/microsoft/CLAP](https://github.com/microsoft/CLAP)
- Wav2Vec2.0 : see [torchaudio official documentation](https://pytorch.org/audio/main/models.html)

Once installed, create a .env configuration file to add paths to the models' checkpoints (an example of such a file can be found [here](./template.env)).

Datasets can be found at the following links :

<table>
<thead>
<tr>
<th>Dataset</th>
<th>Domain</th>
<th>Task</th>
<th>URL</th>
</tr>
</thead>
<tbody>

<tr>
<td>ESC50</td>
<td>Environmental</td>
<td>Classification</td>
<td><a href="https://github.com/karolpiczak/ESC-50">https://github.com/karolpiczak/ESC-50 </a></td>
</tr>

<tr>
<td>Urbansounds8k</td>
<td>Environmental</td>
<td>Classification</td>
<td><a href="https://urbansounddataset.weebly.com/urbansound8k.html">https://urbansounddataset.weebly.com/urbansound8k.html </a></td>

<tr>
<td>FSD-50k</td>
<td>General audio</td>
<td>Multilabel classification</td>
<td><a href="https://zenodo.org/records/4060432">https://zenodo.org/records/4060432 </a></td>
</tr>

<tr>
<td>GTZAN</td>
<td>Music</td>
<td>Genre classification</td>
<td><a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification </a></td>
</tr>

<tr>
<td>NSynth</td>
<td>Musical notes</td>
<td>Pitch estimation</td>
<td><a href="https://magenta.tensorflow.org/datasets/nsynth">https://magenta.tensorflow.org/datasets/nsynth </a></td>
</tr>

<tr>
<td>MagnaTagTune</td>
<td>Music</td>
<td>Music auto-tagging</td>
<td><a href="https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset">https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset </a></td>
</tr>

<tr>
<td>Librispeech</td>
<td>Speech</td>
<td>Automatic speech recognition</td>
<td><a href="https://www.openslr.org/12">https://www.openslr.org/12 </a></td>
</tr>

<tr>
<td>Speech Commands</td>
<td>Speech</td>
<td>Keyword spotting</td>
<td><a href="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz">http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz </a></td>
</tr>

<tr>
<td>Fluent speech commands</td>
<td>Speech</td>
<td>Intent classification</td>
<td><a href="https://groups.google.com/a/fluent.ai/g/fluent-speech-commands">https://groups.google.com/a/fluent.ai/g/fluent-speech-commands </a></td>
</tr>

</tbody>
</table>

## Experiments

Reproducing the results of the paper requires three steps : dataset preparation, training the model, then evaluating it.

### Dataset preparation

To create the lmdb dataset, please run the following script (``` dataset_name ``` should be among (nsynth, gtzan, mtt, esc50, us8k, fsd, libri, commands, fluent))

```shell

python split_to_lmdb --input_path /path/to/dataset_directory --output_path /path/to/lmdb --dataset_name dataset_name --num_cores num_cpu_cores 

```

### Training the model

Configuration is handled with [gin-config](https://github.com/google/gin-config). To run an experiment you will have to provide the configuration files of the foundation model, of the task, and of the experiment :

```bash
python train.py --config configs/models/foundation/clap.gin --config configs/tasks/classif.gin --config configs/experiment/mask.gin 
--run_name path/to/logs --device cuda_device  --db /path/to/lmdb/database --bs batch_size
```

We also provide scripts to reproduce the results of the paper in the scripts/ directory. In this case, just run :

```bash
scripts/experiment_name.sh
```

### Evaluation

tba

## Integration with other models

To use our approach with models that are not in this repository, we provide a tutorial with [this notebook](./notebooks/tutorial.ipynb).