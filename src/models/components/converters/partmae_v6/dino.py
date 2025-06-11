import logging
from src.models.components.partmae_v6 import PARTMaskedAutoEncoderViT


class PARTMaskedAutoEncoderViTFromDINO(PARTMaskedAutoEncoderViT):
    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = state_dict["teacher"]
        # remote .backbone prefix
        state_dict = {
            k.replace("backbone.", ""): v for k, v in state_dict.items()
        }
        # replace head. with dino_head.
        state_dict = {
            k.replace("head.", "dino_head."): v for k, v in state_dict.items()
        }
        # remove all dino_head.last_layer.*
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("dino_head.last_layer.")
        }

        miss, unex = super().load_state_dict(state_dict, strict=False)
        if self.dino_head is None:
            unex = [k for k in unex if not k.startswith("dino_head.")]
        allowed_prefixes = [
            "dino_head.",
            "decoder",
            "mask_pos_token",
            "segment_embed",
            "pose_head.linear.weight",
            "_patch_loss.sigma",
        ]
        other_miss = [
            k for k in miss if not any(k.startswith(p) for p in allowed_prefixes)
        ]
        if other_miss:
            logging.warning("Missing keys: %s", other_miss)
        if unex:
            logging.warning("Unexpected keys: %s", unex)
        if strict:
            assert not other_miss, f"missing keys: {other_miss}"
            assert not unex, f"unexpected keys: {unex}"
        return miss, unex


if __name__ == "__main__":
    # Example usage
    import torch
    model = PARTMaskedAutoEncoderViTFromDINO(embed_dim=384, depth=12, num_heads=6, patch_size=16, pos_embed_mode='learn', ls_init_values=None)
    path = "artifacts/dino_deitsmall16_pretrain_full_checkpoint.pth"
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt)  