import torch
import wandb
from lightning import Callback
from lightning.pytorch.utilities import rank_zero_only
from jaxtyping import Float, Int
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import copy
from torch import Tensor
from src.utils.visualization.visualization import get_transforms_from_reference_patch, create_image_from_transforms, reconstruct_image_from_sampling
from src.models.components.part_vit import PairDiffMLP


class HeadHookLogger(Callback):
    def __init__(self, every_n_steps: int = 100):
        super().__init__()
        self.head_inputs = []
        self.head_outputs = []
        self.every_n_steps = every_n_steps
        self.handle = None

    def _hook_fn(self, module, input, output):
        self.head_input_features = input[0].detach().float().cpu()
        self.head_input_patch_pair_indices = input[1].detach().int().cpu()
        self.head_outputs = output.detach().float().cpu()

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Only register the hook
        if batch_idx % self.every_n_steps == 0:
            self.handle = pl_module.net.head.register_forward_hook(self._hook_fn)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Remove the hook
        if batch_idx % self.every_n_steps == 0 and self.handle is not None:
            self.handle.remove()
            self.handle = None
            fig1 = px.histogram(self.head_input_features.flatten(), title="Head Input Features")
            fig2 = px.histogram(self.head_outputs.flatten(), title="Head Outputs")
            if isinstance(pl_module.net.head, PairDiffMLP):
                # weight distribution
                fig3 = px.histogram(pl_module.net.head.proj.weight.detach().cpu().flatten(), title="PairDiff Weights")
            
            wandb.log({
                'train/head_inputs': fig1,
                'train/head_outputs': fig2,
                'train/head_weights': fig3
            })


# class ReconstructionLogger(Callback):
    # def __init__(self, every_n_steps: int = 100):
        # super().__init__()
        # self.every_n_steps = every_n_steps

    # @rank_zero_only
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # if batch_idx % self.every_n_steps == 0:

            # refpatch_id = self._get_refpatch_id(
                # outputs["patch_positions"], outputs["x_original"].shape[-2:]
            # )
            # ref_transforms = get_transforms_from_reference_patch(
                # refpatch_id, pred_T, patch_pair_indices, patch_positions
            # )

    # def _get_refpatch_id(self, patch_positions: Tensor, image_size: tuple) -> int:
        # H, W = image_size
        # center_y, center_x = H // 2, W // 2
        # distances = ((patch_positions[:, 0] - center_y) ** 2 + 
                    # (patch_positions[:, 1] - center_x) ** 2)
        # return torch.argmin(distances).item()

    # def _log_reconstruction_plots(
        # self,
        # x_original: Tensor,
        # x: Tensor,
        # pred_T: Tensor,
        # patch_positions: Tensor,
        # patch_pair_indices: Tensor,
        # refpatch_id: int,
        # stage: str,
    # ):
        # ref_transforms = get_transforms_from_reference_patch(
            # refpatch_id, pred_T, patch_pair_indices, patch_positions
        # )
        # self._log_reconstruction(
            # refpatch_id,
            # ref_transforms,
            # x_original,
            # x,
            # patch_positions,
            # stage,
        # )

    # def _log_reconstruction(
        # self,
        # refpatch_id: int,
        # transforms,
        # img_raw,
        # img_input,
        # patch_positions,
        # stage: str,
    # ):
        # reconstructed_image = create_image_from_transforms(
            # ref_transforms=transforms,
            # patch_positions=patch_positions,
            # patch_size=self.hparams.patch_size,
            # img=img_raw,
            # refpatch_id=refpatch_id,
        # )

        # sampled_image = reconstruct_image_from_sampling(
            # patch_positions=patch_positions,
            # patch_size=self.hparams.patch_size,
            # img=img_raw,
        # )

        # patch_size = self.hparams.patch_size
        # y, x = patch_positions[refpatch_id]
        # rect = Rectangle(
            # (x, y),
            # patch_size,
            # patch_size,
            # linewidth=1,
            # edgecolor="red",
            # facecolor="none",
        # )

        # fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(img_raw.permute(1, 2, 0))
        # ax[0].set_title("Original Image")
        # ax[0].add_patch(copy(rect))
        # ax[1].imshow(sampled_image.permute(1, 2, 0))
        # ax[1].set_title("Sampled Image")
        # ax[1].add_patch(copy(rect))

        # ax[2].imshow(reconstructed_image.permute(1, 2, 0))
        # ax[2].set_title("Reconstructed Image")
        # ax[2].add_patch(copy(rect))

        # ax[3].imshow(img_input.permute(1, 2, 0))
        # ax[3].set_title("Input Image")
        
        # wandb.log({f"{stage}/reconstruction": wandb.Image(fig)})
        # plt.close(fig)

    # def _log_all_plots(self, output_0, stage: str = "train"):
        # H, W = output_0["x_original"].shape[-2:]
        # patch_size = self.hparams.patch_size
        # refpatch_id = self._get_refpatch_id(output_0["patch_positions"], (H, W))
        # self._log_reconstruction_plots(
            # output_0["x_original"],
            # output_0["x"],
            # output_0["pred_T"],
            # output_0["patch_positions"],
            # output_0["patch_pair_indices"],
            # refpatch_id,
            # stage,
        # )
        # ax[1].set_title("Sampled Image")
        # ax[1].add_patch(copy(rect))

        # ax[2].imshow(reconstructed_image.permute(1, 2, 0))
