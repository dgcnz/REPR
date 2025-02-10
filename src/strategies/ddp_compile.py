from lightning.pytorch.strategies import DDPStrategy
import torch
from torch.nn.parallel import DistributedDataParallel
from typing_extensions import override
import logging


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Don't use DPP + compile unless you know what you are doing
# https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860

class DDPCompileStrategy(DDPStrategy):
    # @override
    # def _setup_model(self, model: torch.nn.Module) -> DistributedDataParallel:
    #     """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
    #     device_ids = self.determine_ddp_device_ids()
    #     log.debug(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
    #     # https://pytorch.org/docs/stable/notes/cuda.html#id5
    #     ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
    #     with ctx:
    #         model =  DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)
    #         model = torch.compile(model)
    #         return model
    def configure_ddp(self):
        super().configure_ddp()
        log.info("Compiling model")
        self.model = torch.compile(self.model)
        log.info("Model compiled")