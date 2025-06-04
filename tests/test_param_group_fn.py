import torch

from src.utils.optimizer import make_param_group_fn


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.decoder_block1 = torch.nn.Linear(8, 4)
        self.decoder_block2 = torch.nn.Linear(4, 4)
        self.decoder_norm = torch.nn.LayerNorm(4)
        self.dino_head = torch.nn.Linear(4, 2)


def _find_group(groups, param):
    for g in groups:
        for p in g["params"]:
            if p is param:
                return g
    raise AssertionError("parameter not found in any group")


def test_param_group_fn_none():
    model = DummyModel()
    pg_fn = make_param_group_fn({}, base_lr=0.1, weight_decay=0.05)
    groups = pg_fn(model)
    assert groups is not None
    g = _find_group(groups, model.conv.weight)
    assert g["lr"] == 0.1 and g["weight_decay"] == 0.05
    g = _find_group(groups, model.conv.bias)
    assert g["lr"] == 0.1 and g["weight_decay"] == 0.0


def test_param_group_fn_assigns_lrs_and_weight_decay():
    model = DummyModel()
    pg_fn = make_param_group_fn(
        {"dino_head": 0.01, "decoder": 0.02}, base_lr=0.1, weight_decay=0.05
    )
    groups = pg_fn(model)
    assert groups is not None

    g = _find_group(groups, model.dino_head.weight)
    assert g["lr"] == 0.01 and g["weight_decay"] == 0.05

    g = _find_group(groups, model.dino_head.bias)
    assert g["lr"] == 0.01 and g["weight_decay"] == 0.0

    g = _find_group(groups, model.decoder_block1.weight)
    assert g["lr"] == 0.02 and g["weight_decay"] == 0.05

    g = _find_group(groups, model.decoder_block1.bias)
    assert g["lr"] == 0.02 and g["weight_decay"] == 0.0

    g = _find_group(groups, model.decoder_block2.weight)
    assert g["lr"] == 0.02 and g["weight_decay"] == 0.05

    g = _find_group(groups, model.decoder_block2.bias)
    assert g["lr"] == 0.02 and g["weight_decay"] == 0.0

    g = _find_group(groups, model.decoder_norm.weight)
    assert g["lr"] == 0.02 and g["weight_decay"] == 0.0

    g = _find_group(groups, model.decoder_norm.bias)
    assert g["lr"] == 0.02 and g["weight_decay"] == 0.0

    g = _find_group(groups, model.conv.weight)
    assert g["lr"] == 0.1 and g["weight_decay"] == 0.05

    g = _find_group(groups, model.conv.bias)
    assert g["lr"] == 0.1 and g["weight_decay"] == 0.0


def test_param_group_fn_no_filter():
    model = DummyModel()
    pg_fn = make_param_group_fn({"dino_head": 0.01}, 0.1, 0.05, filter_bias_and_bn=False)
    groups = pg_fn(model)
    assert groups is not None

    g = _find_group(groups, model.dino_head.bias)
    assert g["lr"] == 0.01 and g["weight_decay"] == 0.05

    g = _find_group(groups, model.conv.bias)
    assert g["lr"] == 0.1 and g["weight_decay"] == 0.05
