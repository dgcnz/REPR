from src.models.components.partmae_v6 import PARTMaskedAutoEncoderViT

class PARTMaskedAutoEncoderViTFromSimDINO(PARTMaskedAutoEncoderViT):
    def load_state_dict(self, state_dict, strict=True):
        state_dict = state_dict["teacher"]

        # 1. Remove the prefix "_orig_mod." from the keys
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        # state_dict.keys()
        # 2. replace prefix head with dino_head
        for k, v in list(state_dict.items()):
            if k.startswith("head."):
                state_dict[k.replace("head.", "dino_head.")] = state_dict.pop(k)

        # 3. replace remove prefix  backbone
        for k, v in list(state_dict.items()):
            if k.startswith("backbone."):
                state_dict[k.replace("backbone.", "")] = state_dict.pop(k)


        miss, unex = super().load_state_dict(state_dict, strict=False)

        # 6. check for missing and unexpected keys
        if self.dino_head is None:
            unex = [k for k in unex if not k.startswith("dino_head.")]

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
            "intra_mask",
            "pos_embed"
        ]
        other_miss = [
            k for k in miss if not any(k.startswith(p) for p in allowed_prefixes)
        ]
        assert len(other_miss) == 0, f"missing keys: {other_miss}"
        return miss, unex


if __name__ == "__main__":
    import torch

    p = "artifacts/vitb16_SimDINOv1_gpu8_bs64_ep100.pth"
    ckpt = torch.load(p)
    model = PARTMaskedAutoEncoderViTFromSimDINO(pos_embed_mode='learn')
    miss, unex = model.load_state_dict(ckpt, strict=True)
    print(f"missing keys: {miss}")
    print(f"unexpected keys: {unex}")
    torch.save({"model": model.state_dict()}, "artifacts/partmae_v6_simdino_untrained.pth")
