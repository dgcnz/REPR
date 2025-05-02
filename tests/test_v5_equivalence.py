import torch
from PIL import Image

# Import the two model factories using aliases so that we can distinguish them.
from src.models.components.partmae_v5 import (
    PART_mae_vit_base_patch16 as partmae_v5_model,
)

from src.models.components.partmae_v5_2 import (
    PART_mae_vit_base_patch16 as partmae_v5_2_model,
)


from src.data.components.transforms.multi_crop_v2 import ParametrizedMultiCropV2


def split_batch(tensor):
    """
    Helper function to split a batched tensor of shape [B, V, ...] into a list
    of V tensors of shape [B, ...].
    """
    return [tensor[:, i, ...] for i in range(tensor.shape[1])]


def test_forward_equivalence():
    gV = 2  # number of global views
    lV = 4  # number of local views
    transform = ParametrizedMultiCropV2(n_global_crops=gV, n_local_crops=lV)
    img = Image.open("artifacts/samoyed.jpg").convert("RGB")
    inputs = transform(img)
    inputs = [x.unsqueeze(0) for x in inputs]  # add batch dimension
    g_x = inputs[0]
    g_params = inputs[1]
    l_x = inputs[2]
    l_params = inputs[3]

    # Instantiate both models.
    torch.manual_seed(42)
    model_v5 = partmae_v5_model(pos_mask_ratio=0.75, mask_ratio=0.75, num_views=gV + lV)
    torch.manual_seed(42)
    model_v5_2 = partmae_v5_2_model(
        pos_mask_ratio=0.75, mask_ratio=0.75, num_views=gV + lV
    )
    model_v5.eval()
    model_v5_2.eval()

    with torch.no_grad():
        torch.manual_seed(42)
        # Forward pass for v5 (expects separate global and local inputs)
        out_v5 = model_v5(g_x, g_params, l_x, l_params)

        # Prepare inputs for v5.1:
        # Split the batched global and local tensors (the 2nd dim holds views) into lists.
        global_list = split_batch(g_x)  # list of length gV, each [B, 3, 224, 224]
        local_list = split_batch(l_x)  # list of length lV, each [B, 3, 96, 96]
        params_global_list = split_batch(g_params)  # list of length gV, each [B, 8]
        params_local_list = split_batch(l_params)  # list of length lV, each [B, 8]

        # Combine global and local views and parameters.
        x_list = global_list + local_list  # total length = gV+lV
        params_list = params_global_list + params_local_list

        # Forward pass for v5.2.
        torch.manual_seed(42)
        out_v52 = model_v5_2(x_list, params_list)


    # Compare selected outputs.
    # Both forward methods return dictionaries that include the total loss and its breakdown,
    # as well as the processed patch positions (after dropping class tokens).
    keys_to_compare = [
        "loss",
        "loss_intra_t",
        "loss_inter_t",
        "loss_intra_s",
        "loss_inter_s",
        "loss_t",
        "loss_s",
    ]
    for key in keys_to_compare:
        assert torch.allclose(
            out_v5[key], out_v52[key], atol=1e-5
        ), f"Mismatch in {key}"

    # Also check that the dropped patch positions match.
    assert torch.allclose(
        out_v5["patch_positions_nopos"], out_v5["patch_positions_nopos"], atol=1e-5
    )


def test_forward_loss_equivalence():
    """
    This test constructs dummy inputs for the internal _forward_loss method.
    The v5 model expects a shapes argument as a 2-tuple,
    while the v5.1 model expects a list of shape tuples.
    We simulate a scenario with two groups (global and local) where:
      - Global: crop size 224, M_g masked tokens per view, and gV views.
      - Local: crop size 96, M_l masked tokens per view, and lV views.
    """
    torch.manual_seed(42)
    B = 2
    gV, M_g = 2, 37
    lV, M_l = 6, 7
    # Total tokens = (gV * M_g) + (lV * M_l)
    T = gV * M_g + lV * M_l

    # Dummy tensors for the loss computation.
    # pose_pred_nopos: predicted transforms [B, T, 4]
    pose_pred_nopos = torch.randn(B, T, 4) * 2
    # patch_pos_nopos: patch positions [B, T, 2]
    patch_pos_nopos_g = torch.randint(0, 224, (B, gV * M_g, 2)) 
    patch_pos_nopos_l = torch.randint(0, 96, (B, lV * M_l, 2))
    patch_pos_nopos = torch.cat((patch_pos_nopos_g, patch_pos_nopos_l), dim=1)
    params = torch.randint(1, 400, (B, T, 4)).float()

    # Shapes arguments:
    shapes = [(224, M_g, gV), (96, M_l, lV)]

    # Instantiate both models.
    torch.manual_seed(42)
    model_v5 = partmae_v5_model(pos_mask_ratio=0.75, mask_ratio=0.75, segment_embed_mode="none")
    torch.manual_seed(42)
    model_v5_2 = partmae_v5_2_model(pos_mask_ratio=0.75, mask_ratio=0.75, segment_embed_mode="none")
    model_v5.eval()

    with torch.no_grad():
        torch.manual_seed(42)
        loss_dict_v5 = model_v5._forward_loss(
            pose_pred_nopos, patch_pos_nopos, params, shapes
        )

        torch.manual_seed(42)
        loss_dict_v5_2 = model_v5_2._forward_loss(
            pose_pred_nopos, patch_pos_nopos, params, shapes
        )

    keys_to_compare = [
        "loss",
        "loss_intra_t",
        "loss_inter_t",
        "loss_intra_s",
        "loss_inter_s",
        "loss_t",
        "loss_s",
    ]
    for key in keys_to_compare:
        assert torch.allclose(
            loss_dict_v5[key], loss_dict_v5_2[key], atol=1e-5
        ), f"Mismatch in {key} for _forward_loss"
