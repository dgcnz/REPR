# conda run --no-capture-output  -n hummingbird python -m scripts.eval model=partmae_v6
# conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
#   model=partmae_v6 \
#   data=voc-mini \
#   'model.pretrained_cfg_overlay.state_dict.state_dict.f="outputs/2025-05-28/11-34-14/epoch_0024.ckpt","outputs/2025-05-28/11-34-14/epoch_0049.ckpt","outputs/2025-05-28/11-34-14/epoch_0074.ckpt","outputs/2025-05-28/11-34-14/epoch_0099.ckpt"'
# HYDRA_FULL_ERROR=1 conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
# model=partmae_v6 \
# data=voc \
# model.pretrained_cfg_overlay.state_dict.state_dict.f=$(ls -d outputs/2025-05-29/12-10-52/*  | grep epoch | paste -sd, -)

import logging
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from hbird.hbird_eval import hbird_evaluation # type: ignore 

import timm  # noqa: F401
from src.utils.io import validate_checkpoint
import lightning as L


def _convnext_patch_size(model: torch.nn.Module) -> int:
    """Return effective patch size for a ConvNeXt model.

    :param model: ConvNeXt backbone.
    :returns: Input patch size mapped to a single token.
    """

    patch = model.stem[0].stride[0]
    for stage in model.stages:
        ds = getattr(stage, "downsample", None)
        if ds is None:
            continue
        for m in ds.modules():
            if isinstance(m, torch.nn.Conv2d):
                patch *= m.stride[0]
    return patch

def _extract_timm_features_vit(model: torch.nn.Module, imgs: torch.Tensor):
    """Return patch tokens from a ViT model.

    :param model: timm vision transformer.
    :param imgs: Input batch of images.
    :returns: Patch tokens and ``None``.
    """

    with torch.no_grad():
        feats = model.forward_features(imgs)
    
    prefix = getattr(model, "num_prefix_tokens", 0)
    return feats[:, prefix:], None


def _extract_timm_features_convnext(model: torch.nn.Module, imgs: torch.Tensor):
    """Return patch tokens from a ConvNeXt model.

    :param model: timm ConvNeXt.
    :param imgs: Input batch of images.
    :returns: Patch tokens and ``None``.
    """

    with torch.no_grad():
        feats = model.forward_features(imgs)

    feats = feats.flatten(2, 3).transpose(1, 2)
    return feats, None


def extract_timm_features(model: torch.nn.Module, imgs: torch.Tensor):
    """Return patch tokens from a timm model."""

    name = model.default_cfg.get("architecture", "")
    if name.startswith("vit_"):
        return _extract_timm_features_vit(model, imgs)
    if name.startswith("convnext_"):
        return _extract_timm_features_convnext(model, imgs)

    raise ValueError(f"Unsupported timm model: {name}")


def attach_artifact_if_missing(ckpt_path: Path, run_id: str, project: str) -> None:
    """Attach ckpt_path as an artifact to the original run if not present."""
    api = wandb.Api()
    entity = "dgcnz"
    run_ref = f"{entity}/{project}/{run_id}" 
    print(run_ref)
    run = api.run(run_ref)
    artifact_name = f"{run_id}-model-{ckpt_path.stem}"
    if any(a.name == artifact_name for a in run.logged_artifacts()):
        logging.warning("%s already logged", artifact_name)
        return
    resume = wandb.init(id=run_id, resume="allow", project=project)
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_reference(f"file://{ckpt_path.resolve()}", name=ckpt_path.name)
    resume.log_artifact(artifact)
    print(f"LOGGED {artifact_name} to {run_ref}")
    resume.finish()



def setup_wandb_logging(cfg: DictConfig):
    """Return a new W&B run and evaluation step."""

    ckpt_mode = cfg.model.ckpt_mode

    if ckpt_mode == "backbone":
        run = wandb.init(project="PART-hummingbird")
        return run, 0
    if ckpt_mode == "checkpoint":
        ckpt_path = Path(cfg.model.ckpt_path)
        ckpt_step, run_id, project, config = validate_checkpoint(ckpt_path)
        # attach_artifact_if_missing(ckpt_path, run_id, project)
        run = wandb.init(
            project="PART-hummingbird",
            group=run_id,
            config=config,
            name=f"{run_id}-{ckpt_step:07d}",
        )
        # run.use_artifact(f"dgcnz/PART-posttrain/{run_id}-model-{ckpt_path.stem}:v0", type="model")
        return run, ckpt_step
    raise ValueError(f"unknown ckpt_mode {ckpt_mode}")


@hydra.main(version_base="1.3", config_path="../fabric_configs/experiment/hummingbird", config_name="config")
def main(cfg: DictConfig):
    run, step = setup_wandb_logging(cfg)
    L.seed_everything(cfg.seed)

    # Build model from config
    model = instantiate(cfg.model, _convert_="all")["net"]
    model = model.eval().to(cfg.device)

    # Extract metadata
    name = model.default_cfg.get("architecture", "")
    if name.startswith("vit_"):
        embed_dim = model.embed_dim
        patch_size = model.patch_embed.proj.weight.shape[-1]
        input_size = model.patch_embed.img_size[-1]
    elif name.startswith("convnext_"):
        input_size = model.default_cfg.get("input_size", (3, 224, 224))[-1]
        embed_dim = model.num_features
        patch_size = _convnext_patch_size(model)
    else:
        raise ValueError(f"Unsupported timm model: {name}")

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
        memory_size=cfg.get("memory_size", None),
        train_fs_path=cfg.data.train_fs,
        val_fs_path=cfg.data.val_fs,
    )
    print(f"mIoU: {mIoU}")

    if run is not None and step is not None:
        run.log({"eval/hbird/mIoU": mIoU}, step=step)
        run.finish()


if __name__ == "__main__":
    main()
