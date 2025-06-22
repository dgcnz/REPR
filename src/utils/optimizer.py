from typing import Callable, Dict, Iterator, List, Optional, Tuple

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

import torch


def _split_wd(
    named_params: Iterator[Tuple[str, torch.nn.Parameter]],
    weight_decay: float,
    lr: float,
) -> Tuple[List[dict], List[List[str]]]:
    """Split parameters into decay and no-decay groups."""

    decayed = []
    decayed_names = []
    no_decay = []
    no_decay_names = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decayed.append(param)
            decayed_names.append(name)
    groups: List[dict] = []
    names: List[List[str]] = []
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr})
        names.append(no_decay_names)
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay, "lr": lr})
        names.append(decayed_names)

    log.debug(
        "Split params into decay=%s and no-decay=%s",
        decayed_names,
        no_decay_names,
    )
    return groups, names


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
            named_list = [(n, p) for n, p in named if p.requires_grad]
            if filter_bias_and_bn:
                new_groups, name_lists = _split_wd(iter(named_list), weight_decay, lr)
                groups.extend(new_groups)
                for g, names in zip(new_groups, name_lists):
                    log.debug(
                        "Added group lr=%s wd=%s params=%s",
                        g["lr"],
                        g["weight_decay"],
                        names,
                    )
            else:
                params = [p for _, p in named_list]
                if params:
                    groups.append({"params": params, "weight_decay": weight_decay, "lr": lr})
                    log.debug(
                        "Added group lr=%s wd=%s params=%s",
                        lr,
                        weight_decay,
                        [n for n, _ in named_list],
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

        log.info("Created %d parameter groups", len(groups))
        for idx, g in enumerate(groups):
            names = [
                n
                for n, p in model.named_parameters()
                if any(p is q for q in g["params"])
            ]
            log.debug(
                "Group %d: lr=%s wd=%s params=%s",
                idx,
                g["lr"],
                g["weight_decay"],
                names,
            )

        return groups

    return param_group_fn


def make_param_group_wd_fn(
    param_groups: Dict[str, Dict[str, float]],
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
            named: Iterator[Tuple[str, torch.nn.Parameter]], lr: float, wd: float
        ) -> None:
            named_list = [(n, p) for n, p in named if p.requires_grad]
            if filter_bias_and_bn:
                new_groups, name_lists = _split_wd(iter(named_list), wd, lr)
                groups.extend(new_groups)
                for g, names in zip(new_groups, name_lists):
                    log.debug(
                        "Added group lr=%s wd=%s params=%s",
                        g["lr"],
                        g["weight_decay"],
                        names,
                    )
            else:
                params = [p for _, p in named_list]
                if params:
                    groups.append({"params": params, "weight_decay": wd, "lr": lr})
                    log.debug(
                        "Added group lr=%s wd=%s params=%s",
                        lr,
                        wd,
                        [n for n, _ in named_list],
                    )

        used = set()
        pending: List[Tuple[List[Tuple[str, torch.nn.Parameter]], float, float]] = []

        for prefix, opts in param_groups.items():
            lr = opts.get("lr")
            if lr is None:
                continue
            wd = opts.get("weight_decay", weight_decay)
            params = [
                (n, p) for n, p in model.named_parameters() if n.startswith(prefix)
            ]
            if not params:
                continue
            used.update({id(p) for _, p in params})
            pending.append((params, lr or base_lr, wd))

        rest = [(n, p) for n, p in model.named_parameters() if id(p) not in used]
        _add_group(rest, base_lr, weight_decay)

        for params, lr, wd in pending:
            _add_group(params, lr, wd)

        log.info("Created %d parameter groups", len(groups))
        for idx, g in enumerate(groups):
            names = [
                n
                for n, p in model.named_parameters()
                if any(p is q for q in g["params"])
            ]
            log.debug(
                "Group %d: lr=%s wd=%s params=%s",
                idx,
                g["lr"],
                g["weight_decay"],
                names,
            )

        return groups

    return param_group_fn
