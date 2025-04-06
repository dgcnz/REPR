import time
import torch
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.utils.events import get_event_storage
from collections import defaultdict

class SimpleTrainerWithAccumulation(SimpleTrainer):
    """
    A SimpleTrainer variant that supports gradient accumulation using a for-loop.
    This approach aggregates multiple mini-batches into a single optimizer update.
    """
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        accumulation_steps=1,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
    ):
        super().__init__(model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward, async_write_metrics)
        self.accumulation_steps = accumulation_steps

    def run_step(self):
        """
        Run a training step that aggregates multiple mini-batches using a for-loop.
        """
        assert self.model.training, "[SimpleTrainerWithAccumulation] model was changed to eval mode!"

        total_loss_dict = defaultdict(float)
        total_data_time = 0.0

        # Zero gradients once at the start of the accumulation cycle.
        self.optimizer.zero_grad()

        for acc_step in range(self.accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start
            total_data_time += data_time

            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

            # Scale the loss to average gradients over accumulation_steps.
            losses = losses / self.accumulation_steps
            losses.backward()
            self.after_backward()

            # Aggregate metrics (scaled accordingly)
            for key, val in loss_dict.items():
                total_loss_dict[key] += val.detach() / self.accumulation_steps

        # Write aggregated metrics.
        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, total_loss_dict, total_data_time, iter=self.iter
            )
        else:
            self._write_metrics(total_loss_dict, total_data_time)

        self.optimizer.step()


class AMPTrainerWithAccumulation(AMPTrainer):
    """
    An AMPTrainer variant that supports gradient accumulation using a for-loop.
    This approach aggregates multiple mini-batches into a single optimizer update with AMP scaling.
    """
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        accumulation_steps=1,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
    ):
        super().__init__(
            model,
            data_loader,
            optimizer,
            gather_metric_period,
            zero_grad_before_forward,
            grad_scaler,
            precision,
            log_grad_scaler,
            async_write_metrics,
        )
        self.accumulation_steps = accumulation_steps

    def run_step(self):
        """
        Run a training step that aggregates multiple mini-batches using a for-loop, with AMP support.
        """
        assert self.model.training, "[AMPTrainerWithAccumulation] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainerWithAccumulation] CUDA is required for AMP training!"

        total_loss_dict = defaultdict(float)
        total_data_time = 0.0

        # Zero gradients once at the start of the accumulation cycle.
        self.optimizer.zero_grad()

        for acc_step in range(self.accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start
            total_data_time += data_time

            with torch.autocast("cuda", dtype=self.precision):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            # Scale the loss to average gradients over accumulation_steps.
            losses = losses / self.accumulation_steps
            self.grad_scaler.scale(losses).backward()
            self.after_backward()

            # Aggregate metrics (scaled accordingly)
            for key, val in loss_dict.items():
                total_loss_dict[key] += val.detach() / self.accumulation_steps

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        # Write aggregated metrics.
        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, total_loss_dict, total_data_time, iter=self.iter
            )
        else:
            self._write_metrics(total_loss_dict, total_data_time)

        # Update model parameters with AMP grad scaling.
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
