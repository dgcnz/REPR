import lightning as L
import wandb
import matplotlib.pyplot as plt
import plotly.express as px

class BaseCallback(L.Callback):
    def on_stage_batch_start(self, trainer, pl_module, batch, batch_idx, stage: str):
        pass
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.on_stage_batch_start(trainer, pl_module, batch, batch_idx, "train")
    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.on_stage_batch_start(trainer, pl_module, batch, batch_idx, "val")
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.on_stage_batch_start(trainer, pl_module, batch, batch_idx, "test")

    def on_stage_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, stage: str):
        pass
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_stage_batch_end(trainer, pl_module, outputs, batch, batch_idx, "train")
    def on_val_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_stage_batch_end(trainer, pl_module, outputs, batch, batch_idx, "val")
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_stage_batch_end(trainer, pl_module, outputs, batch, batch_idx, "test")

    def log_plots(self, plots: dict[str, plt.Figure]):
        wandb.log({name: wandb.Image(fig) for name, fig in plots.items()})
        for fig in plots.values():
            if isinstance(fig, plt.Figure):
                plt.close(fig)
