from lightning.pytorch.strategies import SingleDeviceStrategy
import lightning as L
import torch
import logging


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CompileStrategy(SingleDeviceStrategy):
    def __init__(self, device: str = "cuda:0"):
        super().__init__(device=device)

    def setup(self, trainer: L.Trainer) -> None:
        super().setup(trainer)
        if self.root_device != torch.device("cpu"):
            log.info("Compiling model")
            self.model = torch.compile(self.model)
            log.info("Model compiled")
        else:
            print("Model not compiled because the device is CPU.")
