"""
engine_finetune_det.py

This module implements a Fabric–based Trainer for object detection that:
  • Inherits from detectron2's TrainerBase (so you can register its hooks).
  • Implements all SimpleTrainer features (including full metric aggregation) plus
    gradient accumulation via Fabric.
  • Is designed to work with DDP via Fabric.
"""

import time
import torch
from detectron2.engine.train_loop import SimpleTrainer
from collections import defaultdict


class FabricDetectionTrainer(SimpleTrainer):
    def __init__(
        self,
        fabric,
        model,
        data_loader,
        optimizer,
        accum_iter: int = 1,
        clip_grad: float = 0.0,
        track_grad_norm: bool = False,
        **kwargs,
    ):
        """
        Args:
            fabric: A Lightning Fabric instance.
            model: The detection model.
            data_loader: The training dataloader.
            optimizer: Optimizer for updating the model.
            accum_iter: Number of mini–batches to accumulate gradients over.
            clip_grad: Maximum norm for gradient clipping (0 to disable).
            track_grad_norm: If True, track and log gradient norm.
        """
        super().__init__(model, data_loader, optimizer, **kwargs)
        self.fabric = fabric
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.accum_iter = accum_iter
        self.clip_grad = clip_grad
        self.track_grad_norm = track_grad_norm

    def run_step(self):
        """
        Implements one training iteration:
          • Accumulates gradients over `accum_iter` mini–batches.
          • Uses fabric.no_backward_sync() to reduce communication during accumulation.
          • Performs optimizer step and zeroes gradients.
          • Logs metrics with the full aggregation logic from SimpleTrainer.
        """
        assert self.model.training, (
            "[FabricDetectionTrainer] Model is not in training mode!"
        )

        start_time = time.perf_counter()
        total_loss = 0.0
        loss_dict_agg = defaultdict(float)

        # Accumulate gradients over `accum_iter` mini–batches.
        for i in range(self.accum_iter):
            data = next(self._data_loader_iter)
            # Forward pass: model returns either a tensor or a dict of losses.
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                loss = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                loss = sum(loss_dict.values())

            loss = loss / self.accum_iter  # scale loss for accumulation
            is_last_accum = i == self.accum_iter - 1
            with self.fabric.no_backward_sync(self.model, enabled=not is_last_accum):
                self.fabric.backward(loss)
            total_loss += loss.item()

            # Aggregate loss components using dict.get() to simplify accumulation.
            for k, v in loss_dict.items():
                loss_dict_agg[k] += v.detach() / self.accum_iter

        # Optionally, clip gradients.
        if self.clip_grad > 0:
            self.fabric.clip_gradients(
                self.model, self.optimizer, max_norm=self.clip_grad
            )

        # Optimizer step and zero gradients.
        self.optimizer.step()
        self.optimizer.zero_grad()

        data_time = time.perf_counter() - start_time
        self._write_metrics(loss_dict_agg, data_time)
