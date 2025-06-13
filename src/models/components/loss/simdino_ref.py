import torch
from torch import nn, Tensor
from src.models.components.utils.spectral import batched_logdet
from jaxtyping import Float
from torch.nn import functional as F

class CLSCodingRateLoss(nn.Module):
    # expansion loss for MCR
    def __init__(self, embed_dim: int, gV: int, eps: float, num_chunks: int = 2):
        super().__init__()
        self.eps = eps
        self.gV = gV  # number of global views
        self.num_chunks = num_chunks
        self.register_buffer(
            "I", torch.eye(embed_dim, dtype=torch.float32), persistent=False
        )

    def forward(self, z: Float[Tensor, "B V D"]) -> Tensor:
        """
        :param z: l2 normalized patch embeddings (student)
        """
        B, V, D = z.shape
        # we want to loop over views not batch
        # only use global views for CLS coding rate loss
        z = z.permute(1, 0, 2)[: self.gV]  
        alpha = D / (B * self.eps)
        loss = batched_logdet(self.I, z, alpha, num_chunks=self.num_chunks)
        loss *= (D + B) / (D * B)
        # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps *  B) ** 0.5 / D)
        return {"loss_ccr": -loss}


class CLSInvarianceLoss(nn.Module):
    # compression loss for MCR
    def __init__(self):
        super().__init__()

    def forward(
        self, z_stu: Float[Tensor, "B V D"], z_tea: Float[Tensor, "B V D"]
    ) -> Tensor:
        z_stu = z_stu.permute(1, 0, 2)
        z_tea = z_tea.permute(1, 0, 2)
        sim = F.cosine_similarity(z_tea.unsqueeze(1), z_stu.unsqueeze(0), dim=-1)
        # Trick to fill diagonal
        sim.view(-1, sim.shape[-1])[:: (len(z_stu) + 1), :].fill_(0)
        n_loss_terms = len(z_tea) * len(z_stu) - min(len(z_tea), len(z_stu))
        # Sum the cosine similarities
        comp_loss = sim.mean(2).sum() / n_loss_terms
        return {"loss_cinv": -comp_loss}  # negative because we want to maximize similarity