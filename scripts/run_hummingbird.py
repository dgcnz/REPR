# conda run --no-capture-output  -n hummingbird python -m scripts.eval model=partmae_v6
# conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
#   model=partmae_v6 \
#   data=voc-mini \
#   'model.pretrained_cfg_overlay.state_dict.state_dict.f="outputs/2025-05-28/11-34-14/epoch_0024.ckpt","outputs/2025-05-28/11-34-14/epoch_0049.ckpt","outputs/2025-05-28/11-34-14/epoch_0074.ckpt","outputs/2025-05-28/11-34-14/epoch_0099.ckpt"'
# HYDRA_FULL_ERROR=1 conda run --no-capture-output -n hummingbird python -m scripts.eval --multirun \
# model=partmae_v6 \
# data=voc \
# model.pretrained_cfg_overlay.state_dict.state_dict.f=$(ls -d outputs/2025-05-29/12-10-52/*  | grep epoch | paste -sd, -)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
import re
import torch
import wandb
from hbird.hbird_eval import hbird_evaluation
import timm
from src.utils.io import find_run_id, read_wandb_project_from_config
timm.create_model

def extract_timm_features(model, imgs):
    # Extract the features from the model
    with torch.no_grad():
        features = model.forward_features(imgs)
    # Remove the first token (CLS token)
    return features[:, model.num_prefix_tokens:], None


def setup_wandb_logging(cfg: DictConfig):
    """Resume wandb run and return (run, step) if enabled."""
    if not cfg.get("log_wandb"):
        return None, None

    ckpt_path = Path(cfg.model.pretrained_cfg_overlay.state_dict.state_dict.f)
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_step = int(state["global_step"])
    m = re.search(r"step_(\d+)\.ckpt", ckpt_path.name)
    if m and int(m.group(1)) != ckpt_step:
        raise ValueError(
            f"step mismatch: filename step {m.group(1)} != global_step {ckpt_step}"
        )
    output_dir = ckpt_path.parent
    run_id = find_run_id(output_dir / "wandb")
    project = read_wandb_project_from_config(output_dir / ".hydra" / "config.yaml")
    run = wandb.init(id=run_id, resume="allow", project=project)
    return run, ckpt_step


@hydra.main(version_base="1.3", config_path="../fabric_configs/experiment/hummingbird", config_name="config")
def main(cfg: DictConfig):
    run, step = setup_wandb_logging(cfg)

    # Build model from config
    model = instantiate(cfg.model, _convert_="all")
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
