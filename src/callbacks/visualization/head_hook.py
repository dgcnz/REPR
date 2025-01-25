import torch
import matplotlib.pyplot as plt
import wandb
from lightning.pytorch.utilities import rank_zero_only
from src.models.components.part_vit import PairDiffMLP
from src.utils.misc import clean_tensor, should_log
from src.callbacks.base_callback import BaseCallback


class HeadHookLogger(BaseCallback):
    def __init__(self, every_n_steps: int = -1):
        super().__init__()
        self.head_inputs = []
        self.head_outputs = []
        self.every_n_steps = every_n_steps
        self.handle = None

    def _hook_fn(self, module, input, output):
        self.head_input_features = clean_tensor(input[0])
        # self.head_input_patch_pair_indices = clean_tensor(input[1])
        self.head_outputs = clean_tensor(output)

    @rank_zero_only
    def on_stage_batch_start(self, trainer, pl_module, batch, batch_idx, stage: str):
        # Only register the hook
        if should_log(batch_idx, self.every_n_steps):
            self.handle = pl_module.net.head.register_forward_hook(self._hook_fn)

    @rank_zero_only
    def on_stage_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ):
        if not should_log(batch_idx, self.every_n_steps):
            return

        assert self.handle is not None
        self.handle.remove()
        self.handle = None
        # Create figure 1: Head Input Features histogram
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        axes[0].hist(self.head_input_features.flatten(), bins=50)
        axes[0].set_title("Head Input Features")
        
        # Create figure 2: Head Outputs histogram
        axes[1].hist(self.head_outputs.flatten(), bins=50)
        axes[1].set_title("Head Outputs")
        
        # Create figure 3: PairDiff Weights histogram (if applicable)
        if isinstance(pl_module.net.head, PairDiffMLP):
            axes[2].hist(clean_tensor(pl_module.net.head.proj.weight).flatten(), bins=50)
            axes[2].set_title("PairDiff Weights")
    

        self.log_plots(
            {
                f"{stage}/head_hook": fig,
            }
        )
