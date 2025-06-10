# conda run --no-capture-output  -n hummingbird python -m scripts.eval model=partmae_v6
# conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
#   model=partmae_v6 \
#   data=voc-mini \
#   'model.pretrained_cfg_overlay.state_dict.state_dict.f="outputs/2025-05-28/11-34-14/epoch_0024.ckpt","outputs/2025-05-28/11-34-14/epoch_0049.ckpt","outputs/2025-05-28/11-34-14/epoch_0074.ckpt","outputs/2025-05-28/11-34-14/epoch_0099.ckpt"'
# HYDRA_FULL_ERROR=1 conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
# model=partmae_v6 \
# data=voc \
# model.pretrained_cfg_overlay.state_dict.state_dict.f=$(ls -d outputs/2025-05-29/12-10-52/*  | grep epoch | paste -sd, -)

import os
import re
import logging
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hbird.hbird_eval import hbird_evaluation
import timm  # noqa: F401
from src.utils.io import find_run_id, read_wandb_project_from_config

def extract_timm_features(model, imgs):
    # Extract the features from the model
    with torch.no_grad():
        features = model.forward_features(imgs)
    # Remove the first token (CLS token)
    return features[:, model.num_prefix_tokens:], None


def attach_artifact_if_missing(ckpt_path: Path, run_id: str, project: str) -> None:
    """Attach ckpt_path as an artifact to the original run if not present."""
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY")
    run_ref = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = api.run(run_ref)
    artifact_name = f"model-{ckpt_path.stem}"
    if any(a.name == artifact_name for a in run.logged_artifacts()):
        logging.warning("%s already logged", artifact_name)
        return
    resume = wandb.init(id=run_id, resume="allow", project=project)
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_reference(f"file://{ckpt_path.resolve()}", name=ckpt_path.name)
    resume.log_artifact(artifact)
    resume.finish()


def validate_checkpoint(ckpt_path: Path) -> tuple[int, str, str, dict]:
    """Validate ckpt_path and return (step, run_id, project, config)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint {ckpt_path} not found")
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_step = int(state["global_step"])
    m = re.search(r"step_(\d+)\.ckpt", ckpt_path.name)
    if m and int(m.group(1)) != ckpt_step:
        raise ValueError(
            f"step mismatch: filename step {m.group(1)} != global_step {ckpt_step}"
        )
    output_dir = ckpt_path.parent
    run_id = find_run_id(output_dir / "wandb")
    config_path = output_dir / ".hydra" / "config.yaml"
    project = read_wandb_project_from_config(config_path)
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return ckpt_step, run_id, project, config


def setup_wandb_logging(cfg: DictConfig):
    """Return a new W&B run and evaluation step."""

    ckpt_mode = cfg.model.ckpt_mode

    if ckpt_mode == "backbone":
        run = wandb.init(project="PART-hummingbird")
        return run, 0
    if ckpt_mode == "checkpoint":
        ckpt_path = Path(cfg.model.ckpt_path)
        ckpt_step, run_id, project, config = validate_checkpoint(ckpt_path)
        attach_artifact_if_missing(ckpt_path, run_id, project)
        run = wandb.init(project="PART-hummingbird", group=run_id, config=config)
        run.use_artifact(f"{run_id}/model-{ckpt_path.stem}:latest", type="model")
        return run, ckpt_step
    raise ValueError(f"unknown ckpt_mode {ckpt_mode}")


@hydra.main(version_base="1.3", config_path="../fabric_configs/experiment/hummingbird", config_name="config")
def main(cfg: DictConfig):
    run, step = setup_wandb_logging(cfg)

    # Build model from config
    model = instantiate(cfg.model.net, _convert_="all")
    model = model.eval().to(cfg.device)

    # Extract metadata
    embed_dim = model.embed_dim
    patch_size = model.patch_embed.proj.weight.shape[-1]
    input_size = model.patch_embed.img_size[-1]

    # Run evaluation
    mIoU = hbird_evaluation(
        model,
        d_model=embed_dim,
        patch_size=patch_size,
        batch_size=cfg.batch_size,
        input_size=input_size,
        augmentation_epoch=1,
        device=cfg.device,
        return_knn_details=False,  # whether to return additional NNs details
        nn_method="faiss",
        n_neighbours=30,  # the number of neighbors to fetch per image patch
        nn_params=None,  # Other parameters to be used for the k-NN operator
        ftr_extr_fn=extract_timm_features,
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        memory_size=None,
        train_fs_path=cfg.data.train_fs,
        val_fs_path=cfg.data.val_fs,
    )
    print(f"mIoU: {mIoU}")

    if run is not None and step is not None:
        run.log({"eval/hbird/mIoU": mIoU}, step=step)
        run.finish()


if __name__ == "__main__":
    main()
