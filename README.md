# REPR


## Usage


Use `uv` package manager.

```
uv sync
```

Config files are at `fabric_configs/experiments`.


To run pretraining locally (to debug):

```
uv run python -m scripts.main_pretrain experiment=pretrain/in1k/partmae_v6_vit_b_16/4060ti
```



To run on snellius adjust the following script:
```
sbatch scripts/slurm/train_partmae_v6_h100.sh
```
Model definition is at:

```
src/models/components/partmae_v6.py
```


To run experiments, see (fabric_configs/experiments):

```
uv run python -m src.experiments.linear_classification.main_linear \
        train.seed=0 \
        data=style-imagenette \
        model=partmaev6_b
```