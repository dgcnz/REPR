# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import math
import wandb
import random


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        print("saving model checkpoint debugging...")
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class WeightedMSELoss(torch.nn.modules.Module):
    def __init__(self, distance_weight_mode, mask_prob_weights, num_patches, w_x, w_y, w_w, w_h, device="cpu"):

        self.distance_weight_mode = distance_weight_mode
        self.mask_prob_weights = mask_prob_weights
        self.num_patches = num_patches
        self.device = device
        self.w_x, self.w_y, self.w_w, self.w_h = w_x, w_y, w_w, w_h

        super(WeightedMSELoss, self).__init__()

    def get_weights(self):

        patches_per_width = int(math.sqrt(self.num_patches))
        if self.distance_weight_mode == 0:  # standard MSE
            dists = torch.ones((self.num_patches, self.num_patches), device=self.device)

        elif self.distance_weight_mode == 1:  # loss is weighted by the Euclidean distance of patches
            dists = torch.zeros((self.num_patches, self.num_patches), device=self.device)
            for ref_patch in range(self.num_patches):
                for tgt_patch in range(self.num_patches):
                    x_ref, y_ref = ref_patch % patches_per_width, int(ref_patch / patches_per_width)
                    x_tgt, y_tgt = tgt_patch % patches_per_width, int(tgt_patch / patches_per_width)
                    dists[ref_patch][tgt_patch] = math.sqrt((x_tgt - x_ref)**2 + (y_tgt - y_ref)**2)

            dists = (dists.max() - dists) / dists.max()

        elif self.distance_weight_mode == 2: #loss is weighted by 1-I
            dists = (torch.ones(self.num_patches) - torch.eye(self.num_patches)).to(self.device)
        return dists

    def forward(self, inputs, targets):
        assert targets.shape == inputs.shape
        if self.w_x == 1 and self.w_y == 1 and self.w_w == 1 and self.w_h == 1:
            loss = (targets - inputs) ** 2
        else:
            loss = (self.w_x * (targets[:, 0, :] - inputs[:, 0, :]) ** 2 + self.w_y * (targets[:, 1, :] - inputs[:, 1, :]) ** 2 +
                    self.w_w * (targets[:, 2, :] - inputs[:, 2, :]) ** 2) + self.w_h * (targets[:, 3, :] - inputs[:, 3, :]) ** 2

        if self.mask_prob_weights > 0 or self.distance_weight_mode != 0:
            self.weights = self.get_weights().to(self.device)
            b, c, w, h = loss.shape
            weights = self.weights.repeat(inputs.size(0), c, 1, 1).to(loss.device)
            # Bernoulli(torch.tensor([0.3])): 30% chance 1; 70% chance 0

            # masking points in loss for all c channels
            mask = torch.bernoulli(torch.ones(b, w, h) * (1-self.mask_prob_weights)).repeat(c, 1, 1, 1).permute(1, 0, 2, 3).to(loss.device)
            weights = weights * mask

            #normalizing roughly by the number of 1s in the weights
            # only normalize by the number of unmasked points
            return torch.sum(loss * weights) / torch.count_nonzero(weights)

        return torch.mean(loss)


class SymmetryRegularizer(torch.nn.modules.Module):
    def __init__(self, mask_diag):
        self.mask_diag = mask_diag
        super(SymmetryRegularizer, self).__init__()

    def forward(self, inputs):
        b, c, p1, p2 = inputs.shape
        if inputs.shape[1] == 2:
            assert inputs.shape[2] == inputs.shape[3] and inputs.shape[1] == 2
            # mask_diagonal (if mask_diag=True) because it counts errors twice
            symmetry_loss = inputs + torch.transpose(inputs, dim0=3, dim1=2)
            if self.mask_diag:
                mask = (1 - torch.eye(p1, p2).repeat(b, c, 1, 1)).to(symmetry_loss.device)
                symmetry_loss *= mask
            return torch.mean(symmetry_loss**2)
        elif inputs.shape[1] == 4:  # version 3
            # d_x + (d_x/d_w).T = 0
            # d_y + (d_y/d_h).T = 0
            symmetry_x = inputs[:, 0, :, :] + torch.transpose((inputs[:, 0, :, :]/inputs[:, 2, :, :]), dim0=2, dim1=1)
            symmetry_y = inputs[:, 1, :, :] + torch.transpose((inputs[:, 1, :, :]/inputs[:, 3, :, :]), dim0=2, dim1=1)
            #todo: check why symmetry_x and y are inf: are inputs 0 at any point? (could be because they are outputs)
            symmetry_loss = symmetry_x + symmetry_y
            if self.mask_diag:
                mask = (1 - torch.eye(p1, p2).repeat(b, c, 1, 1)).to(symmetry_loss.device)
                symmetry_loss *= mask
            return torch.mean(symmetry_loss ** 2)
