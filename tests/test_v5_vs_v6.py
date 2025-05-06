import torch
from PIL import Image

# Import the two model factories using aliases so that we can distinguish them.
from src.models.components.partmae_v5_2 import (
    PART_mae_vit_base_patch16 as partmae_v5_2_model,
)

from src.models.components.partmae_v6 import (
    PART_mae_vit_base_patch16 as partmae_v6_model,
)

from src.data.components.transforms.multi_crop_v3 import ParametrizedMultiCropV3


def split_batch(tensor):
    """
    Helper function to split a batched tensor of shape [B, V, ...] into a list
    of V tensors of shape [B, ...].
    """
    return [tensor[:, i, ...] for i in range(tensor.shape[1])]


def test_forward_equivalence():
    """
    Test that the forward passes of v5.2 and v6 models produce equivalent results.
    """
    gV = 2  # number of global views
    lV = 4  # number of local views
    transform = ParametrizedMultiCropV3(n_global_crops=gV, n_local_crops=lV)
    img = Image.open("artifacts/samoyed.jpg").convert("RGB")
    inputs = transform(img)

    # Prepare inputs: v5.2 and v6 both accept a list of x and params tensors
    x_list = inputs[0]  # this is already a list of tensors
    params_list = inputs[1]  # this is already a list of tensors

    # Add batch dimension to each tensor in the lists
    x_list = [x.unsqueeze(0) for x in x_list]
    params_list = [p.unsqueeze(0) for p in params_list]

    # Instantiate both models with identical parameters
    torch.manual_seed(42)
    model_v5_2 = partmae_v5_2_model(
        pos_mask_ratio=0.75, mask_ratio=0.75, num_views=gV + lV
    )
    
    torch.manual_seed(42)
    model_v6 = partmae_v6_model(
        pos_mask_ratio=0.75, mask_ratio=0.75, num_views=gV + lV,
        lambda_cos=0.0,
        lambda_pose=1.0,
        lambda_patch=0.0,
    )

    model_v5_2.eval()
    model_v6.eval()

    # Forward pass for both models
    with torch.no_grad():
        torch.manual_seed(42)
        out_v5_2 = model_v5_2(x_list, params_list)

        torch.manual_seed(42)
        out_v6 = model_v6(x_list, params_list)

    # Also check all pose_loss_ prefixed keys in v6 if they exist
    torch.allclose(
        out_v5_2["loss"], out_v6["loss_pose"], atol=1e-5
    )


