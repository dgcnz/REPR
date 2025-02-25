import torch
import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only
from src.utils.misc import should_log
from src.callbacks.base_callback import BaseCallback

##########################################
# Reconstruction Functions
##########################################

def reconstruct_positions_centered_torch_vectorized(pred_T: torch.Tensor) -> torch.Tensor:
    """
    Vectorized reconstruction of absolute patch positions from pairwise differences.
    
    Args:
        pred_T: Tensor of shape [N, N, 2] containing predicted pairwise differences,
                where ideally pred_T[i, j] ≈ T_j - T_i.
                
    Returns:
        T: Tensor of shape [N, 2] representing the recovered patch positions,
           with a centering constraint (i.e. sum_i T_i = 0).
    """
    N = pred_T.shape[0]
    device = pred_T.device
    
    # Create indices for all i, j pairs.
    idx = torch.arange(N, device=device)
    i_idx, j_idx = torch.meshgrid(idx, idx, indexing='ij')
    i_idx = i_idx.reshape(-1)  # shape: [N*N]
    j_idx = j_idx.reshape(-1)  # shape: [N*N]
    
    valid_mask = i_idx != j_idx
    i_idx = i_idx[valid_mask]  # [N*(N-1)]
    j_idx = j_idx[valid_mask]  # [N*(N-1)]
    num_constraints = i_idx.shape[0]
    
    # Build the system matrix A of shape [2*num_constraints, 2*N].
    A = torch.zeros((2 * num_constraints, 2 * N), device=device)
    
    # Create row indices for vectorized assignment.
    all_rows = torch.arange(2 * num_constraints, device=device)
    # For negative entries: for each pair (i,j), set columns 2*i and 2*i+1.
    neg_cols = torch.stack((2 * i_idx, 2 * i_idx + 1), dim=1).reshape(-1)
    # For positive entries: for each pair (i,j), set columns 2*j and 2*j+1.
    pos_cols = torch.stack((2 * j_idx, 2 * j_idx + 1), dim=1).reshape(-1)
    
    # Fill in the matrix: -1 for the i side, +1 for the j side.
    A[all_rows, neg_cols] = -1.0
    A[all_rows, pos_cols] = 1.0
    
    # Build the right-hand side vector b from pred_T.
    pred_T_flat = pred_T.reshape(-1, 2)  # [N*N, 2]
    b = pred_T_flat[valid_mask].reshape(-1)  # [2*num_constraints]
    
    # Add centering constraint: sum_i T_i = 0 (for x and y).
    centering_A = torch.zeros((2, 2 * N), device=device)
    centering_A[0, 0::2] = 1.0  # x coordinates
    centering_A[1, 1::2] = 1.0  # y coordinates
    A = torch.cat([A, centering_A], dim=0)          # [2*num_constraints+2, 2*N]
    b = torch.cat([b, torch.zeros(2, device=device)], dim=0)  # [2*num_constraints+2]
    
    # Solve the least-squares system.
    solution = torch.linalg.lstsq(A, b).solution  # shape: [2*N]
    T = solution.view(N, 2)
    return T

def reconstruct_image_from_sampling(patch_positions: torch.Tensor, patch_size: int, img: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct an image by pasting patches at their sampled positions.
    
    Args:
        patch_positions: Tensor of shape [N, 2] with the top-left coordinates of each patch.
        patch_size: Size (height/width) of each square patch.
        img: Original image tensor of shape [C, H, W].
        
    Returns:
        A reconstructed image tensor of shape [C, H, W].
    """
    C, H, W = img.shape
    canvas = torch.zeros_like(img)
    count_map = torch.zeros((1, H, W), device=img.device)
    N = patch_positions.shape[0]
    
    for i in range(N):
        y, x = patch_positions[i].round().long().tolist()
        # Ensure the patch lies within image boundaries.
        y = max(0, min(H - patch_size, y))
        x = max(0, min(W - patch_size, x))
        patch = img[:, y:y+patch_size, x:x+patch_size]
        canvas[:, y:y+patch_size, x:x+patch_size] += patch
        count_map[:, y:y+patch_size, x:x+patch_size] += 1
    
    count_map[count_map == 0] = 1
    return canvas / count_map

def reconstruct_image_from_model_outputs(
    patch_positions_vis: torch.Tensor,
    ids_remove_pos: torch.Tensor,
    pred_T: torch.Tensor,
    original_image: torch.Tensor,
    patch_size: int
) -> torch.Tensor:
    """
    Reconstruct an image using the model outputs for one sample.
    
    Args:
        patch_positions_vis: Tensor of shape [N, 2] with the ground-truth patch coordinates.
        ids_remove_pos: Tensor of shape [N_nopos] with indices of patches that did not get a position embedding.
        pred_T: Tensor of shape [(N_nopos)**2, 2] with predicted pairwise differences for masked patches.
        original_image: Tensor of shape [C, H, W] (the original image).
        patch_size: The patch size (assumed square).
    
    Returns:
        A reconstructed image tensor of shape [C, H, W] using the predicted patch positions.
    """
    C, H, W = original_image.shape
    N = patch_positions_vis.shape[0]
    N_nopos = ids_remove_pos.shape[0]
    
    # Reshape pred_T to [N_nopos, N_nopos, 2] and recover absolute positions for masked patches.
    pred_T_reshaped = pred_T.view(N_nopos, N_nopos, 2)
    T_pred = reconstruct_positions_centered_torch_vectorized(pred_T_reshaped)  # [N_nopos, 2]
    
    # Align T_pred to the coordinate system of the ground-truth positions.
    if N_nopos > 0:
        gt_masked = patch_positions_vis[ids_remove_pos]
        offset = gt_masked.float().mean(dim=0)
        T_pred = T_pred + offset
    
    # Extract patch images from the original image using patch_positions_vis.
    patches = []
    for i in range(N):
        y, x = patch_positions_vis[i]
        y = int(round(y.item()))
        x = int(round(x.item()))
        y = max(0, min(H - patch_size, y))
        x = max(0, min(W - patch_size, x))
        patch = original_image[:, y:y+patch_size, x:x+patch_size]
        patches.append(patch)
    patches = torch.stack(patches, dim=0)  # [N, C, patch_size, patch_size]
    
    # Build the full positions by replacing positions for indices in ids_remove_pos with T_pred.
    full_positions = patch_positions_vis.clone()  # [N, 2]
    for j, patch_idx in enumerate(ids_remove_pos.tolist()):
        full_positions[patch_idx] = T_pred[j]
    
    # Paste patches onto a canvas.
    canvas = torch.zeros_like(original_image)
    count_map = torch.zeros((1, H, W), device=original_image.device)
    for i in range(N):
        pos = full_positions[i]
        y, x = pos.round().long().tolist()
        y = max(0, min(H - patch_size, y))
        x = max(0, min(W - patch_size, x))
        patch = patches[i]
        canvas[:, y:y+patch_size, x:x+patch_size] += patch
        count_map[:, y:y+patch_size, x:x+patch_size] += 1
    count_map[count_map == 0] = 1
    return canvas / count_map

##########################################
# Lightning Callback
##########################################

class ReconstructionLogger(BaseCallback):
    def __init__(
        self, every_n_steps: int = -1, num_samples: int = 1, every_n_epochs: int = 1
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def _plot(
        self,
        pl_module,
        img,
        patch_positions_vis,
        pred_T,
        patch_pair_indices,  # not used here, kept for consistency
        ids_remove_pos,
    ):
        # We plot 3 columns:
        #  Column 0: Original Image
        #  Column 1: Ground Truth Reconstruction (from sampling)
        #  Column 2: Predicted Reconstruction (using model outputs)
        fig, axes = plt.subplots(
            self.num_samples, 3, figsize=(15, 5 * self.num_samples)
        )
        if self.num_samples == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            # Column 0: Original image.
            ax[0].set_title("Original Image")
            ax[0].imshow(img[i].permute(1, 2, 0).cpu().numpy())
            ax[0].axis("off")

            # Column 1: Ground Truth Reconstruction (using sampled patch positions).
            reconstruction_gt = reconstruct_image_from_sampling(
                patch_positions=patch_positions_vis[i],
                patch_size=pl_module.net.patch_size,
                img=img[i],
            )
            ax[1].set_title("Ground Truth Reconstruction")
            ax[1].imshow(reconstruction_gt.permute(1, 2, 0).cpu().numpy())
            ax[1].axis("off")

            # Column 2: Predicted Reconstruction (using model outputs for masked patches).
            reconstruction_pred = reconstruct_image_from_model_outputs(
                patch_positions_vis=patch_positions_vis[i],
                ids_remove_pos=ids_remove_pos[i],
                pred_T=pred_T[i],
                original_image=img[i],
                patch_size=pl_module.net.patch_size,
            )
            ax[2].set_title("Predicted Reconstruction")
            ax[2].imshow(reconstruction_pred.permute(1, 2, 0).cpu().numpy())
            ax[2].axis("off")
        plt.tight_layout()
        return fig

    @rank_zero_only
    def on_stage_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ):
        if not should_log(
            batch_idx, self.every_n_steps, trainer.current_epoch, self.every_n_epochs
        ):
            return
        fig_rec = self._plot(
            pl_module,
            batch["image"][: self.num_samples].detach().cpu(),
            outputs["patch_positions_vis"][: self.num_samples].detach().cpu(),
            outputs["pred_T"][: self.num_samples].detach().cpu(),
            outputs["patch_pair_indices"][: self.num_samples].detach().cpu(),
            outputs["ids_remove_pos"][: self.num_samples].detach().cpu(),
        )
        self.log_plots({f"{stage}/reconstruction": fig_rec})
