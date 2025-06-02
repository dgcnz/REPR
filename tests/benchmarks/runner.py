from statistics import mean, median, stdev
import pytest
import torch
from .memtracker import MemTracker
from typing import Callable


def get_args_kwargs(args, kwargs, args_gen, kwargs_gen):
    if args_gen:
        args = args_gen()
    if kwargs_gen:
        kwargs = kwargs_gen()
    return args, kwargs


class Runner(object):
    def __init__(self, request):
        self.request = request
        self.metric_optim = dict()

    def benchmark(
        self,
        func,
        args: tuple = tuple(),
        kwargs: dict = dict(),
        args_gen: Callable[[], tuple] = None,
        kwargs_gen: Callable[[], dict] = None,
        n_warmup: int = 0,
        n_runs: int = 2,
    ):
        if bool(args_gen) and bool(args):
            raise ValueError(
                "Only one of args_gen or args can be provided, not both"
            )
        # assert kwargs_gen or kwargs, "kwargs_gen or kwargs must be provided"
        # kwargs_gen and kwargs are optional, but not both can be provided
        assert not (
            kwargs_gen and kwargs
        ), "Only one of kwargs_gen or kwargs can be provided"

        self._before_iter()
        for _ in range(n_warmup):
            args, kwargs = get_args_kwargs(args, kwargs, args_gen, kwargs_gen)
            func(*args, **kwargs)

        self._after_warmup()

        events = []
        for i in range(n_runs):
            event = {}
            args, kwargs = get_args_kwargs(args, kwargs, args_gen, kwargs_gen)
            self._before_run(event)
            func(*args, **kwargs)
            self._after_run(event)
            events.append(event)
        self._after_iter(events)
        stats = self._gather_stats(events) or {}
        self.request.node.user_properties.extend(stats.items())
        self.request.node.user_properties.append(("n_runs", n_runs))
        self.request.node.user_properties.append(("metric_optim", self.metric_optim))

    def group_by(self, group_name: str):
        # check that the group is not already in the user properties, otherwise throw error
        for key, _ in self.request.node.user_properties:
            if key == "group":
                raise ValueError("Group already exists in user properties")
        self.request.node.user_properties.append(("group", group_name))

    def drop_columns(self, exclude: list[str]):
        self.request.node.user_properties.append(("drop_columns", exclude))

    def highlight_top_k(self, k: int):
        self.request.node.user_properties.append(("highlight_top_k", k))

    def sort_by(self, metric_name: str):
        if metric_name not in self.metric_optim:
            raise ValueError(f"Metric {metric_name} not in metric_optim")
        self.request.node.user_properties.append(("sort_by", metric_name))

    def _after_warmup(self):
        pass

    def _before_run(self, event: dict):
        pass

    def _after_run(self, event: dict):
        pass

    def _after_iter(self, events: list[dict]):
        pass

    def _gather_stats(self, events: list[dict]) -> dict:
        pass


class CUDARunner(Runner):
    def __init__(self, request):
        super().__init__(request)
        self.mem_tracker = MemTracker()
        self.metric_optim = {
            "time/mean (ms)": "min",
            "time/median (ms)": "min",
            "time/stdev": "min",
            "time/min (ms)": "min",
            "time/max (ms)": "min",
            "mem/max_reserved (MB)": "min",
            "mem/max_used (MB)": "min",
            "mem/max_tracked (MB)": "min",
        }

    def _before_iter(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _before_run(self, event: dict):
        free, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
        event["start_time"] = torch.cuda.Event(enable_timing=True)
        event["start_time"].record()
        event["start_mem_used"] = total - free

    def _after_run(self, event: dict):
        event["end_time"] = torch.cuda.Event(enable_timing=True)
        event["end_time"].record()
        free, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
        event["end_mem_used"] = total - free

    def _after_warmup(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.mem_tracker.start_polling()

    def _after_iter(self, events: list[dict]):
        torch.cuda.synchronize()
        self.mem_tracker.stop_polling()
        events[-1]["max_memory_reserved"] = torch.cuda.max_memory_reserved() / 1024**2
        events[-1]["max_tracked"] = self.mem_tracker.get_max_mem()

    def _gather_stats(self, events: list[dict]) -> dict:
        times = []
        max_mem = 0
        for event in events:
            start = event["start_time"]
            end = event["end_time"]
            times.append(start.elapsed_time(end))
            max_mem = max(max_mem, event["start_mem_used"])
            max_mem = max(max_mem, event["end_mem_used"])

        time_stats = {
            "time/mean (ms)": mean(times),
            "time/median (ms)": median(times),
            "time/stdev": stdev(times),
            "time/min (ms)": min(times),
            "time/max (ms)": max(times),
        }
        mem_stats = {
            "mem/max_reserved (MB)": events[-1]["max_memory_reserved"],
            "mem/max_used (MB)": max_mem / 1024**2,
            "mem/max_tracked (MB)": events[-1]["max_tracked"],
        }
        return {**time_stats, **mem_stats}


@pytest.fixture
def benchmark_v2(request):
    runner = CUDARunner(request)
    return runner
