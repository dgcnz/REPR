# https://github.com/RobinWu218/SimDINO/blob/8149fc4ed8ee3fd97b7bde900447414e6669e71b/simdino/main_dino.py#L497-L559
# https://arxiv.org/abs/2502.10385
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as dist_nn


class MCRLoss(nn.Module):
    def __init__(self, ncrops, reduce_cov=0, expa_type=0, eps=0.5, coeff=1.0):
        super().__init__()
        self.ncrops = ncrops
        self.eps = eps
        self.coeff = coeff
        self.reduce_cov = reduce_cov
        self.expa_type = expa_type

    def forward(self, student_feat, teacher_feat):
        """
        Expansion Loss and Compression Loss between features of the teacher and student networks.
        """
        student_feat = student_feat.view(self.ncrops, -1, student_feat.shape[-1])
        teacher_feat = teacher_feat.view(2, -1, teacher_feat.shape[-1])

        comp_loss = self.calc_compression(student_feat, teacher_feat)
        if self.expa_type == 0:  # only compute expansion on global views
            expa_loss = self.calc_expansion(student_feat[: len(teacher_feat)])
        elif self.expa_type == 1:
            expa_loss = self.calc_expansion(
                (student_feat[: len(teacher_feat)] + teacher_feat) / 2
            )
        loss = -self.coeff * comp_loss - expa_loss
        return loss, comp_loss.detach(), expa_loss.detach()

    def calc_compression(self, student_feat_list, teacher_feat_list):
        """
        Compute compression loss between student and teacher features.
        """
        # Convert lists of tensors to a single tensor for vectorized operations

        sim = F.cosine_similarity(
            teacher_feat_list.unsqueeze(1), student_feat_list.unsqueeze(0), dim=-1
        )
        sim.view(-1, sim.shape[-1])[:: (len(student_feat_list) + 1), :].fill_(
            0
        )  # Trick to fill diagonal

        n_loss_terms = len(teacher_feat_list) * len(student_feat_list) - min(
            len(teacher_feat_list), len(student_feat_list)
        )
        # Sum the cosine similarities
        comp_loss = sim.mean(2).sum() / n_loss_terms
        # global_comp_loss = (sim[:, :len(teacher_feat_list)].mean(2).sum()).detach_().div_(len(teacher_feat_list))
        return comp_loss

    def calc_expansion(self, feat_list) -> torch.Tensor:
        """
        Compute expansion loss using Coding Rate estimation.
        """
        cov_list = []
        num_views = len(feat_list)
        m, p = feat_list[0].shape

        cov_list = [W.T.matmul(W) for W in feat_list]
        cov_list = torch.stack(cov_list)
        N = 1
        if dist.is_initialized():
            N = dist.get_world_size()
            if self.reduce_cov == 1:
                cov_list = dist_nn.all_reduce(cov_list)
        scalar = p / (m * N * self.eps)
        I = torch.eye(p, device=cov_list[0].device)
        loss: torch.Tensor = 0
        for i in range(num_views):
            loss += (
                torch.linalg.cholesky_ex(I + scalar * cov_list[i])[0]
                .diagonal()
                .log()
                .sum()
            )
        loss /= num_views
        loss *= (
            (p + N * m) / (p * N * m)
        )  # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * N * m) ** 0.5 / p)
        return loss
