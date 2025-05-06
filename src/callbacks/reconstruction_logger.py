from typing import Any, Dict, Tuple, List
import torch
from lightning import Fabric
import wandb
import matplotlib.pyplot as plt
from torch import Tensor
import torch.utils._pytree as pytree
from itertools import chain
from src.utils.visualization.reconstruction_v5_anchor_reparam import reconstruction_lstsq_with_anchor_reparam
from src.utils.visualization.reconstruction_v5_gt import reconstruction_gt


@torch.no_grad()
def preprocess_reconstructions(
    batch: Tuple[Any, Any],
    out: Dict[str, Any],
    num_samples: int,
    device: str
) -> Dict[str, Any]:
    """Prepare and truncate raw batch/outputs to IO-ready dict."""
    x = list(zip(*batch[0]))[:num_samples]
    patch_positions = out["patch_positions_nopos"][:num_samples]
    pred_dT = out["pred_dT"][:num_samples]
    crop_params = [
        [p[4:8] for p in params] for params in zip(*batch[1])
    ][:num_samples]
    num_tokens = list(chain.from_iterable(
        [[n_tokens] * n_views for (_, n_tokens, n_views) in out["shapes"]]
    ))
    io: Dict[str, Any] = {
        "x": x,
        "patch_positions_nopos": patch_positions,
        "pred_dT": pred_dT,
        "crop_params": crop_params,
        "num_tokens": num_tokens,
    }
    return pytree.tree_map_only(Tensor, lambda t: t.detach().to(device), io)


@torch.no_grad()
def compute_reconstructions(
    io: Dict[str, Any],
    patch_size: int = 16,
    canonical_img_size: int = 224,
    max_scale_ratio: float = 5.97,
) -> List[Tuple[Tensor, Tensor]]:
    """Compute GT and predicted reconstruction tensors from preprocessed IO."""
    reconstructions: List[Tuple[Tensor, Tensor]] = []
    for idx, x in enumerate(io["x"]):
        gt = reconstruction_gt(
            x=x,
            patch_positions_nopos=io["patch_positions_nopos"][idx],
            num_tokens=io["num_tokens"],
            crop_params=io["crop_params"][idx],
            patch_size=patch_size,
            canonical_img_size=canonical_img_size,
        )
        pred, *_ = reconstruction_lstsq_with_anchor_reparam(
            x=x,
            patch_positions_nopos=io["patch_positions_nopos"][idx],
            num_tokens=io["num_tokens"],
            crop_params=io["crop_params"][idx],
            patch_size=patch_size,
            canonical_img_size=canonical_img_size,
            max_scale_ratio=max_scale_ratio,
            pred_dT=io["pred_dT"][idx],
        )
        reconstructions.append((gt, pred))
    return reconstructions


def plot_reconstructions(
    reconstructions: List[Tuple[Tensor, Tensor]]
) -> plt.Figure:
    """Plot GT vs. predicted reconstructions in a grid."""
    n = len(reconstructions)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(8, 4 * n),
        constrained_layout=True,
    )
    if n == 1:
        axes = axes[None, :]
    for idx, (gt, pred) in enumerate(reconstructions):
        ax_gt, ax_pred = axes[idx]
        ax_gt.imshow(gt.permute(1, 2, 0).cpu())
        if idx == 0:
            ax_gt.set_title("GT Reconstruction")
        ax_gt.axis("off")
        ax_pred.imshow(pred.permute(1, 2, 0).cpu())
        if idx == 0:
            ax_pred.set_title("Pred Reconstruction")
        ax_pred.axis("off")
    return fig


class ReconstructionLogger:
    """Callback for logging image reconstructions every N training steps."""
    def __init__(
        self,
        every_n_steps: int = 1,
        num_samples: int = 4,
        device: str = "cuda",
    ):
        if every_n_steps < 1:
            raise ValueError(f"every_n_steps must be >= 1, got {every_n_steps}")
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.every_n_steps = every_n_steps
        self.num_samples = num_samples
        self.device = device

    def on_train_batch_end(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        batch: Tuple[Any, Any],
        outputs: Dict[str, Any],
        batch_idx: int,
        global_step: int,
        **kwargs,
    ) -> None:
        if batch_idx % self.every_n_steps == 0:
            self._log(fabric, model, batch, outputs, global_step)

    def _log(
        self,
        fabric: Fabric,
        model: torch.nn.Module,
        batch: Tuple[Any, Any],
        outputs: Dict[str, Any],
        global_step: int,
    ) -> None:
        # Preprocess IO
        io = preprocess_reconstructions(batch, outputs, self.num_samples, self.device)
        # Compute reconstructions
        reconstructions = compute_reconstructions(io, model)
        # Plot and log
        fig = plot_reconstructions(reconstructions)
        if fabric.is_global_zero:
            wandb.log({"reconstruction": wandb.Image(fig)}, step=global_step)
        plt.close(fig)
