import torch
from torch import Tensor


def clean_tensor(t: Tensor):
    t = t.detach()
    if torch.is_floating_point(t):
        t = t.float()
    return t.cpu()


def should_log(batch_idx: int, every_n_steps: int) -> bool:
    return batch_idx == 0 or (every_n_steps > 0 and batch_idx % every_n_steps != 0)
