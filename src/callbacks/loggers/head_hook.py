import matplotlib.pyplot as plt
from lightning.pytorch.utilities import rank_zero_only
from src.models.components.part_vit import PairDiffMLP
from src.utils.misc import clean_tensor, should_log
from src.callbacks.base_callback import BaseCallback


class HeadHookLogger(BaseCallback):
    def __init__(self, every_n_steps: int = -1):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.handles = []

        # Hook data
        self.head_input_features = None
        self.head_outputs = None
        self.model_output = None

    def _head_hook_fn(self, module, input, output):
        self.head_input_features = clean_tensor(input[0])
        # self.head_input_patch_pair_indices = clean_tensor(input[1])
        self.head_outputs = clean_tensor(output)
    
    def _model_hook_fn(self, module, input, output):
        # rescale by logit_scale to [-1, 1]
        self.model_output = clean_tensor(output / module.logit_scale) 

    def _attach_hooks(self, pl_module):
        self.handles =[
            pl_module.net.head.register_forward_hook(self._head_hook_fn),
            pl_module.net.register_forward_hook(self._model_hook_fn)
        ]

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Only register the hook
        if should_log(batch_idx, self.every_n_steps):
            self._attach_hooks(pl_module)

    def _detach_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _clean_hook_data(self):
        self.head_input_features = None
        self.head_outputs = None
        self.model_output = None

    def _plot(self, axes: plt.Axes, pl_module, _hist_kwargs: list[dict] | dict = dict()) :
        _hist_kwargs = _hist_kwargs or {} # if [] or None, default to {}
        if isinstance(_hist_kwargs, dict):
            _hist_kwargs = [_hist_kwargs] * 4
        assert len(_hist_kwargs) == 4

        axes[0].hist(self.head_input_features.flatten(), bins=50, **_hist_kwargs[0])
        std = self.head_input_features.std().item()
        axes[0].set_title(f"Head Input Features | std: {std:.2f}")
        
        # Create figure 2: Head Outputs histogram
        axes[1].hist(self.head_outputs.flatten(), bins=50, **_hist_kwargs[1])
        std = self.head_outputs.std().item()
        axes[1].set_title(f"Head Outputs | std: {std:.2f}")

        
        # Create figure 3: PairDiff Weights histogram (if applicable)
        if isinstance(pl_module.net.head, PairDiffMLP):
            axes[2].hist(clean_tensor(pl_module.net.head.proj.weight).flatten(), bins=50, **_hist_kwargs[2])
            std = pl_module.net.head.proj.weight.std().item()
            axes[2].set_title(f"PairDiff Weights | std: {std:.2f}")


        # Create figure 4: Model Output histogram
        axes[3].hist(self.model_output.flatten(), bins=50, **_hist_kwargs[3])
        std = self.model_output.std().item()
        axes[3].set_title(f"Model Output | std: {std:.2f}")



    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not should_log(batch_idx, self.every_n_steps):
            return
        assert self.handles
        self._detach_hooks()

        fig, axes = plt.subplots(4, 1, figsize=(10, 20))
        self._plot(axes, pl_module)

        self.log_plots(
            {
                f"train/head_hook": fig,
            }
        )
