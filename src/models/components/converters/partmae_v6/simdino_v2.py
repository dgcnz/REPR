import re
from src.models.components.partmae_v6 import PARTMaskedAutoEncoderViT
"""
def process_simdinov2(ckpt):
    ckpt = ckpt["teacher"]
    ckpt = {
        k.removeprefix("backbone."): v
        for k, v in ckpt.items()
        if k.startswith("backbone")
    }
    # revert chunk weight
    ckpt = {
        re.sub(r"blocks\.(\d+)\.(\d+)\.", r"blocks.\2.", k): v for k, v in ckpt.items()
    }
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2", patch_size=16, img_size=224, num_classes=0, pretrained=False
    )
    # ckpt = timm.models.vision_transformer.checkpoint_filter_fn(ckpt, model)

    # model = timm.create_model("vit_large_patch14_reg4_dinov2", patch_size=16, img_size=224, num_classes=0)
    # model.load_state_dict(ckpt)
    return ckpt

"""

"""
def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict
"""

"""
There's a couple of caveats which makes this non-trivial:

First, the dinov2 is trained with 
pos_embed torch.Size([1, 197, 768])
but then timm does
torch.Size([1, 196, 768])
because it removes the class token pos embed and directly adds it to the cls_token.

also the register token in reg_token is used in timm and we're not using it in our model.

so it's not trivial. :(
"""



class PARTMaskedAutoEncoderViTFromSimDINOv2(PARTMaskedAutoEncoderViT):
    def load_state_dict(self, state_dict, strict=True):
        # 1. load teacher
        state_dict = state_dict["teacher"]

        # 2. remove prefix  backbone
        for k, v in list(state_dict.items()):
            if k.startswith("backbone."):
                state_dict[k.replace("backbone.", "")] = state_dict.pop(k)

        # 3. revert chunk weight
        state_dict = {
            re.sub(r"blocks\.(\d+)\.(\d+)\.", r"blocks.\2.", k): v
            for k, v in state_dict.items()
        }

        # 5. remove mask_token
        state_dict.pop("mask_token", None)

        miss, unex = super().load_state_dict(state_dict, strict=False)
        # 6. check for missing and unexpected keys
        if self.dino_head is None:
            unex = [k for k in unex if not k.startswith("dino_head.")]

        if "register_tokens" in unex:
            raise ValueError("register_tokens not set in this model.")
        if "blocks.1.ls1.gamma" in unex:
            raise ValueError("LayerScale missing. Please set ls_init_values.")

        # no missing keys
        assert len(unex) == 0, f"unexpected keys: {unex}"

        # filter unexpected keys
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
        assert len(other_miss) == 0, f"missing keys: {other_miss}"
        return miss, unex


if __name__ == "__main__":
    import torch

    p = "artifacts/vitb16_reg4_SimDNIOv2_ep100.pth"
    ckpt = torch.load(p)
    model = PARTMaskedAutoEncoderViTFromSimDINOv2(num_register_tokens=4, ls_init_values=1e-5)
    miss, unex = model.load_state_dict(ckpt, strict=False)
    print(f"missing keys: {miss}")
    print(f"unexpected keys: {unex}")
