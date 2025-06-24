def preprocess_base(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def preprocess(state_dict):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # pop mask_token
    state_dict.pop("mask_token", None)
    return state_dict


if __name__ == "__main__":
    import torch
    import timm
    from functools import partial

    from src.models.components.droppos import DropPos_mae_vit_base_patch16

    p = "artifacts/DropPos_pretrain_vit_base_patch16.pth"
    state_dict = torch.load(p, map_location="cpu")

    timm_model = timm.create_model(
        "vit_base_patch16_224",
        num_classes=0,
        pretrained=False,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    )


    timm_model.load_state_dict(preprocess(state_dict), strict=True)
    timm_model.eval()

    og = DropPos_mae_vit_base_patch16(multi_task=False)
    miss, unex = og.load_state_dict(preprocess_base(state_dict), strict=False)
    # print(f"Missed keys: {miss}")
    # print(f"Unexpected keys: {unex}")
    og.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        timm_out = timm_model.forward_features(x)
        og_x = og.patch_embed(x)
        og_x_after_pe = og_x + og.pos_embed[:, 1:, :].data.repeat(og_x.shape[0], 1, 1)
        og_x_with_cls = torch.cat(
            (og.cls_token + og.pos_embed[:, :1, :].expand(og_x.shape[0], -1, -1), og_x_after_pe),
            dim=1,
        )
        og_z = og_x_with_cls.clone()
        for blk in og.blocks:
            og_z = blk(og_z)
        og_out = og.norm(og_z)


    assert og_out.shape == timm_out.shape, f"Output shapes do not match: {og_out.shape} vs {timm_out.shape}"
    assert torch.allclose(og_out, timm_out, atol=1e-5, rtol=1e-5)
    print("Outputs match successfully!")
    # print magnitude/norm of output
    print(f"Output norm: {og_out.norm().item()}")
    print(f"timm model norm: {timm_out.norm().item()}")


