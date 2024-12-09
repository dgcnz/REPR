import torch
import timm
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
import math
import timeit
import lightning as L
import logging
from typing import Any


def sample_with_replacement(n: int, k: int) -> Float[Tensor, "k"]:
    return torch.randint(n, (k,))


def sample_without_replacement(n: int, k: int) -> Float[Tensor, "k"]:
    return torch.randperm(n)[:k]


class PARTViT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_pairs: int,
        embed_dim: int = 384,
        head_type: str = "pairwise_mlp",
        num_targets: int = 2,  # 2: [dx, dy] 4: [dx, dy, dw, dh]
    ):
        super().__init__()
        self.backbone = backbone
        self.n_pairs = n_pairs
        if head_type == "pairwise_mlp":
            self.head = nn.Linear(2 * embed_dim, num_targets)
        else:
            raise NotImplementedError(f"Head type {head_type} not implemented")

    def forward(self, x: Float[Tensor, "b c h w"]):
        x: Float[Tensor, "b n d"] = self.backbone(x)
        _, n, _ = x.size()
        assert (
            math.isqrt(n) ** 2 == n
        ), f"If the number of patches is not a perfect square, the cls token might be sneaking in. n={n}"

        # Sample n_pairs by first sampling i and then sampling j, output: (b, n_pairs, 2)
        idx = sample_with_replacement(n, self.n_pairs)
        jdx = sample_with_replacement(n, self.n_pairs)

        z: Float[Tensor, "b n_pairs 2 d"] = torch.stack([x[:, idx], x[:, jdx]], dim=-2)
        # concat the two patches
        z: Float[Tensor, "b n_pairs 2*d"] = z.view(z.size(0), z.size(1), -1)
        z = self.head(z)

        return z


class PARTModule(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cli_logger = logging.getLogger(self.__class__.__name__)

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._compile = compile

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self._compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


vit: timm.models.VisionTransformer = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=0,
    global_pool="",
    class_token=False,
)

embed_dim = vit.embed_dim
print(embed_dim)
model = PARTViT(vit, n_pairs=6, embed_dim=vit.embed_dim).cuda()
x = torch.randn(1, 3, 224, 224).cuda()
print(model(x).shape)
compiled_model = torch.compile(model)
print(compiled_model(x).shape)

N_ITER = 100

time_normal = timeit.timeit(lambda: model(x), number=N_ITER)
time_compiled = timeit.timeit(lambda: compiled_model(x), number=N_ITER)
print(f"Normal: {time_normal}")
print(f"Compiled: {time_compiled}")

# print(vit(x).shape)
# out = vit.forward_features(x)
# print(out.shape)
# vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, features_only=True, out_indices=(-1,))
# out = vit(x)
# print(len(out))
# print(out[0].shape)
