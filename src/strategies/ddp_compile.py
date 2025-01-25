from lightning.pytorch.strategies import DDPStrategy
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

class DDPCompileStrategy(DDPStrategy):
    def configure_ddp(self):
        super().configure_ddp()
        self.model: DDP = torch.compile(self.model)
        
