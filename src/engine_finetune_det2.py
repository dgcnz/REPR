import time
import torch
from detectron2.engine import SimpleTrainer, AMPTrainer
from detectron2.utils.events import  get_event_storage

class SimpleTrainerWithAccumulation(SimpleTrainer):
    """
    A SimpleTrainer variant that supports gradient accumulation.
    The gradients are accumulated for a fixed number of iterations (accumulation_steps)
    before an optimizer update is performed.
    """
    def __init__(self, model, data_loader, optimizer, accumulation_steps=1, **kwargs):
        """
        Args:
            model: a torch.nn.Module.
            data_loader: an iterable data loader.
            optimizer: a torch optimizer.
            accumulation_steps (int): number of iterations to accumulate gradients.
            **kwargs: additional keyword arguments passed to the base SimpleTrainer.
        """
        super().__init__(model, data_loader, optimizer, **kwargs)
        self.accumulation_steps = accumulation_steps
        self._accumulation_counter = 0

    def run_step(self):
        """
        Run a single training step with gradient accumulation.
        """
        assert self.model.training, "[SimpleTrainerWithAccumulation] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # Only zero gradients at the start of an accumulation cycle.
        if self.zero_grad_before_forward:
            if self._accumulation_counter == 0:
                self.optimizer.zero_grad()

        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            loss = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            loss = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            if self._accumulation_counter == 0:
                self.optimizer.zero_grad()

        # Scale loss to average gradients over accumulation_steps.
        loss = loss / self.accumulation_steps
        loss.backward()
        self._accumulation_counter += 1

        self.after_backward()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        # Only update weights and reset the counter when the accumulation step is complete.
        if self._accumulation_counter == self.accumulation_steps:
            self.optimizer.step()
            self._accumulation_counter = 0



class AMPTrainerWithAccumulation(AMPTrainer):
    """
    AMPTrainer variant that supports gradient accumulation.
    Gradients are accumulated for a specified number of iterations before performing an optimizer update.
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
        self._accumulation_counter = 0

    def run_step(self):
        """
        Run a single training step with AMP and gradient accumulation.
        """
        assert self.model.training, "[AMPTrainerWithAccumulation] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainerWithAccumulation] CUDA is required for AMP training!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # Zero gradients only once per accumulation cycle.
        if self.zero_grad_before_forward:
            if self._accumulation_counter == 0:
                self.optimizer.zero_grad()

        with torch.autocast("cuda", dtype=self.precision):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            if self._accumulation_counter == 0:
                self.optimizer.zero_grad()

        # Scale loss to average gradients over accumulation_steps.
        losses = losses / self.accumulation_steps
        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        self._accumulation_counter += 1

        # Only update weights and reset counter when accumulation is complete.
        if self._accumulation_counter == self.accumulation_steps:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self._accumulation_counter = 0
