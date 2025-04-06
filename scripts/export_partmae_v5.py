import argparse
import timm
import torch
from pathlib import Path
import math



def process_state_dict(state_dict):
    """Process the state dict by reshaping and removing decoder parts."""
    # Reshape patch embedding weights
    C = 3
    hidden_dim = state_dict["patch_embed.proj.weight"].shape[0]
    patch_size = math.isqrt(state_dict["patch_embed.proj.weight"].shape[1] // C)
    print(f"Patch size: {patch_size}")
    print(f"Hidden dim: {hidden_dim}")
    state_dict["patch_embed.proj.weight"] = state_dict[
        "patch_embed.proj.weight"
    ].reshape(hidden_dim, C, patch_size, patch_size)
    
    # Filter out decoder parts
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder_")}
    
    # Remove unnecessary keys
    for key in ["mask_pos_token", "segment_embed"]:
        if key in state_dict:
            state_dict.pop(key)
            
    return state_dict


def validate_backbone(state_dict):
    """Validate that the state dict loads correctly into a ViT backbone."""
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
    )
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    
    # Log missing and unexpected keys for debugging
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    # Validation checks
    assert len(unexpected) == 0, f"Unexpected keys found: {unexpected}"
    assert len(missing) == 2, f"Expected 2 missing keys, got {len(missing)}: {missing}"
    assert "head.weight" in missing, "head.weight should be missing"
    assert "head.bias" in missing, "head.bias should be missing"


def get_backbone_path(ckpt_path):
    """Generate the output path for the backbone checkpoint."""
    path = Path(ckpt_path)
    epoch = path.stem.split("_")[-1]
    return path.parent / f"backbone_{epoch}.ckpt"


def main():
    parser = argparse.ArgumentParser(
        description="Export PartMAE model to backbone checkpoint"
    )
    parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    # Load, process, and validate the state dict
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = ckpt["model"]
    state_dict = process_state_dict(state_dict)
    validate_backbone(state_dict)

    # Save the backbone state dict
    backbone_path = get_backbone_path(args.ckpt_path)
    torch.save(state_dict, backbone_path)
    print(f"Backbone saved to {backbone_path.resolve()}")


if __name__ == "__main__":
    main()
