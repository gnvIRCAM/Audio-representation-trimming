# Audio representation trimming

Repository for the paper "Keep what you need : extracting efficient subnetworks from large audio representation models", by David Genova, Philippe Esling, and Tom Hurlin.

## Experiments

Reproducing the results of the paper requires three steps : dataset preparation, training the model, then evaluating it.

### Dataset preparation

To create the lmdb dataset, please run the following script (```dataset_name``` should be among (nsynth, gtzan, mtt, esc50, us8k, fsd, libri, commands, fluent))

```shell

python split_to_lmdb --input_path /path/to/dataset_directory --output_path /path/to/lmdb --dataset_name dataset_name --num_cores num_cpu_cores 

```

### Training the model

tba

### Evaluation 

tba

## Integration with other models

To use our approach with models that are not in this repository, we provide a tutorial with the notebook tba.ipynb
