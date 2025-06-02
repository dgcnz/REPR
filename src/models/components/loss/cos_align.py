from torch import nn, Tensor


class CosineAlignmentLoss(nn.Module):
    """
    TODO: verify:
    did some stationarity analysis and it seems that this
    helps avoiding pose_head.linear=0 solution
    not sure if i did it right

    currently setting its coefficient to 0
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_dT: Tensor, gt_dT: Tensor) -> Tensor:
        # pred_dT, gt_dT: [B, T, T, C]
        B, _, _, C = pred_dT.shape
        # flatten the pair dims → [B, P, C], where P=T*T
        pred = pred_dT.view(B, -1, C)
        gt = gt_dT.view(B, -1, C)
        # normalize each vector to unit length
        pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + self.eps)
        gt_norm = gt / (gt.norm(dim=-1, keepdim=True) + self.eps)
        # cosine similarity per pair
        cos_sim = (pred_norm * gt_norm).sum(dim=-1)  # [B, P]
        # loss = 1 − cos_sim, then mean over all pairs and batch
        # this is the same as -cos_sim but easier to interpret
        return (1 - cos_sim).mean()