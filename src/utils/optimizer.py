from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch


def _split_wd(
    named_params: Iterator[Tuple[str, torch.nn.Parameter]],
    weight_decay: float,
    lr: float,
) -> List[dict]:
    """Split parameters into decay and no-decay groups."""

    decayed = []
    no_decay = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decayed.append(param)
    groups: List[dict] = []
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr})
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay, "lr": lr})
    return groups


def make_param_group_fn(
    param_groups: Dict[str, Optional[float]],
    base_lr: float,
    weight_decay: float,
    filter_bias_and_bn: bool = True,
) -> Callable[[torch.nn.Module], List[dict]]:
    """Return a callable for ``timm.create_optimizer_v2``.

    Parameters whose names start with one of the keys in ``param_groups`` are
    collected into a separate parameter group using that key's learning rate.
    The remaining parameters use ``base_lr``. These "rest" parameters are placed
    at the start of the resulting list so ``optimizer.param_groups[0]`` always
    corresponds to the base learning rate. If ``filter_bias_and_bn`` is ``True``
    (default), biases and 1D tensors (typically norms) are placed in a
    weight-decay-free group similar to
    :func:`timm.optim.param_groups_weight_decay`.
    """

    def param_group_fn(model: torch.nn.Module) -> List[dict]:
        groups: List[dict] = []

        def _add_group(
            named: Iterator[Tuple[str, torch.nn.Parameter]], lr: float
        ) -> None:
            if filter_bias_and_bn:
                groups.extend(_split_wd(named, weight_decay, lr))
            else:
                params = [p for _, p in named if p.requires_grad]
                if params:
                    groups.append(
                        {"params": params, "weight_decay": weight_decay, "lr": lr}
                    )

        used = set()
        pending: List[Tuple[List[Tuple[str, torch.nn.Parameter]], float]] = []

        for prefix, lr in param_groups.items():
            if lr is None:
                continue
            params = [
                (n, p) for n, p in model.named_parameters() if n.startswith(prefix)
            ]
            if not params:
                continue
            used.update({id(p) for _, p in params})
            pending.append((params, lr or base_lr))

        rest = [(n, p) for n, p in model.named_parameters() if id(p) not in used]
        _add_group(rest, base_lr)

        for params, lr in pending:
            _add_group(params, lr)

        return groups

    return param_group_fn
