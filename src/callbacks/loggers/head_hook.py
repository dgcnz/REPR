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
        self.handles = []

    def _head_hook_fn(self, module, input, output):
        self.head_input_features = clean_tensor(input[0])
        # self.head_input_patch_pair_indices = clean_tensor(input[1])
        self.head_outputs = clean_tensor(output)
    
    def _model_hook_fn(self, module, input, output):
        # rescale by logit_scale to [-1, 1]
        self.model_output = clean_tensor(output) / module.logit_scale

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Only register the hook
        if should_log(batch_idx, self.every_n_steps):
            self.handles =[
                pl_module.net.head.register_forward_hook(self._head_hook_fn),
                pl_module.net.register_forward_hook(self._model_hook_fn)
            ]

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not should_log(batch_idx, self.every_n_steps):
            return
        
        assert self.handles
        for handle in self.handles:
            handle.remove()
        self.handles = []
        # Create figure 1: Head Input Features histogram
        fig, axes = plt.subplots(4, 1, figsize=(10, 20))
        axes[0].hist(self.head_input_features.flatten(), bins=50)
        axes[0].set_title("Head Input Features")
        
        # Create figure 2: Head Outputs histogram
        axes[1].hist(self.head_outputs.flatten(), bins=50)
        axes[1].set_title("Head Outputs")
        
        # Create figure 3: PairDiff Weights histogram (if applicable)
        if isinstance(pl_module.net.head, PairDiffMLP):
            axes[2].hist(clean_tensor(pl_module.net.head.proj.weight).flatten(), bins=50)
            axes[2].set_title("PairDiff Weights")

        # Create figure 4: Model Output histogram
        axes[3].hist(self.model_output.flatten(), bins=50)
        axes[3].set_title("Model Output")

        self.log_plots(
            {
                f"train/head_hook": fig,
            }
        )
