from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import timm
import torch
from huggingface_hub.utils import validate_repo_id
from omegaconf import OmegaConf


def load_config(config_path: Path) -> dict:
    """Load backbone configuration from yaml file."""
    cfg = OmegaConf.load(config_path)
    version = next(tag for tag in cfg.tags if tag.startswith("partmae_"))
    
    return {
        "name": "vit_base_patch16_224",
        "channels": cfg.model.in_chans,
        "patch_size": cfg.model.patch_size,
        "img_size": cfg.model.img_size,
        "version": version,
        "keys_to_remove": ["mask_pos_token", "segment_embed"],
        "expected_missing": ["head.weight", "head.bias"]
    }


def process_state_dict(state_dict: Dict[str, torch.Tensor], config: dict) -> Dict[str, torch.Tensor]:
    """
    Process the state dict by reshaping and removing decoder parts.
    TESTED ONLY FOR version v5.2
    """
    hidden_dim = state_dict["patch_embed.proj.weight"].shape[0]
    state_dict["patch_embed.proj.weight"] = state_dict["patch_embed.proj.weight"].reshape(
        hidden_dim, config["channels"], config["patch_size"], config["patch_size"]
    )
    
    return {k: v for k, v in state_dict.items() 
            if not k.startswith("decoder_") and k not in config["keys_to_remove"]}


def get_repo_id(config: dict) -> str:
    """Generate repository ID from model config."""
    patch_size = config["patch_size"]
    img_size = config["img_size"]
    version = config["version"]
    
    model_name = f"vit_base_patch{patch_size}_{img_size}"
    return f"{model_name}.{version}"


def validate_backbone(state_dict: Dict[str, torch.Tensor], config: dict) -> None:
    """Validate that the state dict loads correctly into a ViT backbone."""
    backbone = timm.create_model(config["name"], pretrained=False)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    
    if missing or unexpected:
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    
    assert not unexpected, f"Unexpected keys found: {unexpected}"
    assert set(missing) == set(config["expected_missing"]), \
        f"Mismatched missing keys, got: {missing}"


def get_backbone_path(ckpt_path: str | Path) -> Path:
    """Generate the output path for the backbone checkpoint."""
    path = Path(ckpt_path)
    epoch = path.stem.split("_")[-1]
    return path.parent / f"backbone_{epoch}.ckpt"


def push_to_hub(
    state_dict: Dict[str, torch.Tensor],
    config: dict,
) -> None:
    """Push the backbone model to the Hugging Face Hub."""
    repo_id = get_repo_id(config)
    validate_repo_id(repo_id)
    
    model = timm.create_model(config["name"], pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    
    timm.models.push_to_hf_hub(
        model, repo_id, safe_serialization=False
    )
    print(f"Model pushed to Hugging Face Hub: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PartMAE model to backbone checkpoint")
    parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--no-save", action="store_true", help="Don't save the model locally")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    config = load_config(ckpt_path.parent / "config.yaml")
    
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = process_state_dict(ckpt["model"], config)
    validate_backbone(state_dict, config)

    if not args.no_save:
        backbone_path = get_backbone_path(args.ckpt_path)
        torch.save(state_dict, backbone_path)
        print(f"Backbone saved to {backbone_path.resolve()}")
    
    push_to_hub(state_dict, config)


if __name__ == "__main__":
    main()
