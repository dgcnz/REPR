import torch

from src.callbacks.common.metric_logger import MetricLogger


class DummyFabric:
    def __init__(self):
        self.logged = []
        self.is_global_zero = True

    def log_dict(self, metrics, step=None):
        self.logged.append((step, metrics))


def test_log_lr_single_group():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = MetricLogger()
    fabric = DummyFabric()
    logger._log_lr(fabric, 1, opt)
    assert fabric.logged == [(1, {"train/lr": 0.1})]


def test_log_lr_multiple_groups():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD([
        {"params": [model.weight], "lr": 0.1},
        {"params": [model.bias], "lr": 0.01},
    ])
    logger = MetricLogger()
    fabric = DummyFabric()
    logger._log_lr(fabric, 2, opt)
    assert fabric.logged == [(2, {"train/lr": 0.1, "train/lr1": 0.01})]

