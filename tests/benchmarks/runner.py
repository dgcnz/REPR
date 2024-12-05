from statistics import mean, median, stdev
import pytest
import torch
from .memtracker import MemTracker


class Runner(object):
    def __init__(self, request):
        self.request = request

    def benchmark(
        self,
        func,
        args: tuple = tuple(),
        kwargs: dict = dict(),
        n_warmup: int = 0,
        n_runs: int = 2,
    ):
        self._before_iter()
        for _ in range(n_warmup):
            func(*args, **kwargs)

        self._after_warmup()

        events = []
        for _ in range(n_runs):
            event = {}
            self._before_run(event)
            func(*args, **kwargs)
            self._after_run(event)
            events.append(event)
        self._after_iter(events)
        stats = self._gather_stats(events) or {}
        self.request.node.user_properties.extend(stats.items())

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
        events[-1]["max_memory_reserved"] = torch.cuda.max_memory_reserved() / 1024 ** 2
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
            "time/mean": mean(times),
            "time/median": median(times),
            "time/stdev": stdev(times),
            "time/min": min(times),
            "time/max": max(times),
        }
        mem_stats = {
            "mem/max_reserved (MB)": events[-1]["max_memory_reserved"],
            "mem/max_used (MB)": max_mem / 1024 ** 2,
            "mem/max_tracked (MB)": events[-1]["max_tracked"],
        }
        return {**time_stats, **mem_stats}


@pytest.fixture
def benchmark_v2(request):
    runner = CUDARunner(request)
    return runner