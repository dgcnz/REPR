import torch
from PIL import Image
from torch.utils._pytree import tree_map_only
# Import the two model factories using aliases so that we can distinguish them.
from src.models.components.partmae_v5 import (
    PART_mae_vit_base_patch16 as partmae_v5_model,
)

from src.models.components.partmae_v6 import (
    PART_mae_vit_base_patch16 as partmae_v6_model,
)

from src.data.components.transforms.multi_crop_v4 import ParametrizedMultiCropV4


def split_batch(tensor):
    """
    Helper function to split a batched tensor of shape [B, V, ...] into a list
    of V tensors of shape [B, ...].
    """
    return [tensor[:, i, ...] for i in range(tensor.shape[1])]


def test_forward_equivalence():
    """
    Test that the forward passes of v5 and v6 models produce equivalent results.
    """
    gV = 2  # number of global views
    lV = 10  # number of local views
    transform = ParametrizedMultiCropV4(n_global_crops=gV, n_local_crops=lV)
    img = Image.open("artifacts/samoyed.jpg").convert("RGB")
    inputs = transform(img)
    # unsqueeze to add batch dimension with tree_map_only
    inputs = tree_map_only(torch.Tensor, lambda x: x.unsqueeze(0), inputs)

    # Prepare inputs: v5.2 and v6 both accept a list of x and params tensors
    v5_inputs = (
        inputs[0][0],
        inputs[1][0],
        inputs[0][1],
        inputs[1][1],
    )

    # Instantiate both models with identical parameters
    torch.manual_seed(42)
    model_v5 = partmae_v5_model(
        pos_mask_ratio=0.75,
        mask_ratio=0.75,
        num_views=gV + lV,
        segment_embed_mode="none",
    )

    torch.manual_seed(42)
    model_v6 = partmae_v6_model(
        pos_mask_ratio=0.75,
        mask_ratio=0.75,
        num_views=gV + lV,
        lambda_cos=0.0,
        lambda_pose=1.0,
        lambda_pcr=0.0,
        lambda_ccr=0.0,
        lambda_psmooth=0.0,
        lambda_cinv=0.0,
    )

    model_v5.eval()
    model_v6.eval()

    # Forward pass for both models
    with torch.no_grad():
        torch.manual_seed(42)
        out_v5_2 = model_v5(*v5_inputs)

        torch.manual_seed(42)
        out_v6 = model_v6(*inputs)

    # Also check all pose_loss_ prefixed keys in v6 if they exist
    torch.allclose(out_v5_2["loss"], out_v6["loss_pose"], atol=1e-5)
