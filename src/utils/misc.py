import torch
from torch import Tensor


def clean_tensor(t: Tensor):
    t = t.detach()
    if torch.is_floating_point(t):
        t = t.float()
    return t.cpu()


def should_log(batch_idx: int, every_n_steps: int, current_epoch: int = 0, every_n_epochs: int = 1) -> bool:
    if current_epoch % every_n_epochs == 0:
        return batch_idx == 0 or (every_n_steps > 0 and batch_idx % every_n_steps != 0)
    return False
