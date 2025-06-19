import math

def preprocess(state_dict):
    # Reshape patch embedding weights
    state_dict = state_dict["model"]
    if "register_tokens" in state_dict:
        # this triggers the convert_dinov2 in timm's loading checkpoint
        # which is needed for correctly parsing the positional embeddings
        state_dict["mask_token"] = state_dict["mask_pos_token"]

    C = 3
    D = state_dict["patch_embed.proj.weight"].shape[0] 
    P = math.isqrt(state_dict["patch_embed.proj.weight"].shape[1] // C)
    state_dict["patch_embed.proj.weight"] = state_dict["patch_embed.proj.weight"].reshape(
        D, C, P, P
    )
    # state_dict["patch_embed.proj.weight"] = state_dict["patch_embed.proj.weight"].permute(0, 2, 3, 1)

    # Filter out decoder parts
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder_")}

    # Remove unnecessary keys
    for key in list(state_dict.keys()):
        for to_delete in ["mask_pos_token", "segment_embed", "pose_head", "dino_head"]:
            if key.startswith(to_delete):
                state_dict.pop(key, None)
    return state_dict


if __name__ == "__main__":
    from src.models.components.partmae_v5_2 import PARTMaskedAutoEncoderViT
    from timm.models import VisionTransformer
    import timm
    import torch
    ckpt_path = "outputs/2025-04-11/10-15-18/epoch_0199.ckpt"

    # Example usage
    I = 224
    P = 16
    src = PARTMaskedAutoEncoderViT(
        sampler="ongrid_canonical",
        mask_ratio=0,
        pos_mask_ratio=0,
        # pos_embed_mode="sincos", # also works for "learn"
        img_size=I,
        patch_size=P,
        # norm_layer=torch.nn.LayerNorm,
    )
    state_dict = preprocess({"model": src.state_dict()})

    tgt: VisionTransformer = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=I,
        patch_size=P,
        norm_layer=torch.nn.LayerNorm,
        
    )
    tgt.load_state_dict(state_dict, strict=True)

    x = torch.randn(1, 3, I, I)
    tgt_out = dict()
    src_out = dict()

    tgt_out["x_tok"] = tgt.patch_embed(x)
    src_out.update(dict(zip(["x_tok", "patch_positions_vis"], src.patch_embed(x))))
    assert torch.allclose(tgt_out["x_tok"], src_out["x_tok"], atol=1e-5), (
        "Patch embeddings do not match!"
    )

    #tgt_out["x_tok_pos"] = tgt._pos_embed(tgt_out["x_tok"])
    # src_out.update(
    #     dict(
    #         zip(
    #             ["x_tok_pos", "ids_rem", "ids_res"],
    #             src._pos_embed(src_out["x_tok"], src_out["patch_positions_vis"]),
    #         )
    #     )
    # )

    # pos_embed, *_ = src._get_pos_embed(src_out["patch_positions_vis"])
    # assert torch.allclose(pos_embed, tgt.pos_embed[:, 1:], atol=1e-5), (
    #     "Positional embeddings do not match!"
    # )  # fails here!!
    # assert torch.allclose(src.cls_pos_embed, tgt.pos_embed[:, 0], atol=1e-5), (
    #     "Class token positional embeddings do not match!"
    # )
    # assert torch.allclose(src_out["x_tok_pos"], tgt_out["x_tok_pos"], atol=1e-5), (
    #     "pre transformer doesn not match!"
    # )
    # src_out["x_prepared"], *_ = src.prepare_tokens(x)
    # assert torch.allclose(src_out["x_prepared"], tgt_out["x_tok_pos"], atol=1e-5), (
    #     "pre transformer doesn not match!"
    # )
    src_out["z"] = src.forward_encoder(x)["z_enc"]
    tgt_out["z"] = tgt.forward_features(x)
    # assert shapes match first
    print(src_out["z"].shape, tgt_out["z"].shape)
    assert src_out["z"].shape == tgt_out["z"].shape, (
        f"Shapes do not match! src: {src_out['z'].shape}, tgt: {tgt_out['z'].shape}"
    )
    # then check values
    assert torch.allclose(src_out["z"], tgt_out["z"], atol=1e-5), (
        "Outputs do not match!"
    )
