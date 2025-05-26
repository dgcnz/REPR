# PROFILING: https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
from src.utils import pylogger
import torch
from torch import nn

log = pylogger.RankedLogger(__name__)


class CUDAProfiler(object):
    def __init__(self, warmup_steps: int = 10, profile_steps: int = 5):
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.step = 0

    def on_train_start(self, model: nn.Module, **kwargs):
        autonvtx(model)

    def on_train_batch_start(self, **kwargs):
        if self.step == self.warmup_steps:
            torch.cuda.cudart().cudaProfilerStart()
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_push("iteration{}".format(self.step))
        self.step += 1

    def on_train_forward_start(self, **kwargs):
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_push("forward")

    def on_train_forward_end(self, **kwargs):
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_pop()

    def on_train_backward_start(self, **kwargs):
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_push("backward")

    def on_train_backward_end(self, **kwargs):
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_pop()

    def on_train_optimizer_step_start(self, **kwargs):
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_push("opt.step()")

    def on_train_optimizer_step_end(self, **kwargs):
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_pop()

    def on_train_batch_end(self, **kwargs):
        if self.step >= self.warmup_steps:
            torch.cuda.nvtx.range_pop()

        if self.step == self.warmup_steps + self.profile_steps:
            torch.cuda.cudart().cudaProfilerStop()
            log.info("CUDA Profiler stopped")
            exit(0)


# https://github.com/zasdfgbnm/autonvtx/blob/master/autonvtx/__init__.py
def autonvtx(model, name=None):
    if name is None:
        name = type(model).__name__
    else:
        name = name + ": " + type(model).__name__

    def push(*args, _name=name, **kwargs):
        torch.cuda.nvtx.range_push(_name)

    def pop(*args, **kwargs):
        torch.cuda.nvtx.range_pop()

    model.register_forward_pre_hook(push)
    model.register_forward_hook(pop)

    for name, child in model.named_children():
        autonvtx(child, name)

    return model
