import torch
from torch import nn, Tensor
from functools import partial
from src.models.components import partmae_v5
from itertools import groupby, accumulate


def _list(n: int):
    return [None] * n


class PARTMaskedAutoEncoderViT(partmae_v5.PARTMaskedAutoEncoderViT):
    r"""
    A generalized variant that accepts a list of crops and corresponding params.
    Each crop (and param) is assumed to be ordered contiguously by resolution.
    For example, if x[0] and x[1] are high resolution (global) and x[2:] are low resolution (local),
    then the total number of views must equal self.num_views.
    """

    def _forward_loss(
        self,
        pose_pred_nopos: Tensor,
        patch_pos_nopos: Tensor,
        params: Tensor,
        shapes: list,
    ) -> dict:
        # Note: mostly unchanged from v5's _forward_loss, entirely compatible.
        B = pose_pred_nopos.shape[0]
        device = pose_pred_nopos.device

        # Build labels list in Python to avoid data-dependent tensor creation.
        # resgroup_idx  is the resolution group index
        # in_resgroup_idx is the index of the view within that group
        resgroup_idx, in_resgroup_idx = torch.tensor(
            [
                (i, j)
                for i, (_, M, V) in enumerate(shapes)
                for j in range(V)  # each view in that group
                for _ in range(M)  # one entry per patch
            ],
            device=device,
        ).unbind(1)

        # Build intra- and inter-view masks
        # mask tells use which patches belong to the same view
        mask = resgroup_idx[:, None] == resgroup_idx[None, :]
        mask = mask & (in_resgroup_idx[:, None] == in_resgroup_idx[None, :])
        mask = mask.float()[None, ..., None].expand(B, -1, -1, 1)
        diag, offdiag = mask.sum(), (1 - mask).sum()

        # Build token crop sizes: for each group, create a (M*V, 1) tensor filled with the crop size.
        token_crop_sizes = torch.cat(
            [torch.full((M * V, 1), cs, device=device) for cs, M, V in shapes]
        )[None].expand(B, -1, -1)

        # Compute pairwise differences.
        pred_dT = self._compute_pred(pose_pred_nopos)
        gt_dT = self._compute_gt(patch_pos_nopos, params, token_crop_sizes)
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        # Compute loss components for translation and scale.
        loss_intra_t = (loss_full[..., :2] * mask).sum() / diag
        loss_inter_t = (loss_full[..., :2] * (1 - mask)).sum() / offdiag
        loss_intra_s = (loss_full[..., 2:] * mask).sum() / diag
        loss_inter_s = (loss_full[..., 2:] * (1 - mask)).sum() / offdiag
        loss_t = self.alpha_t * loss_inter_t + (1 - self.alpha_t) * loss_intra_t
        loss_s = self.alpha_s * loss_inter_s + (1 - self.alpha_s) * loss_intra_s
        loss = self.alpha_ts * loss_t + (1 - self.alpha_ts) * loss_s

        return {
            "loss_intra_t": loss_intra_t,
            "loss_inter_t": loss_inter_t,
            "loss_intra_s": loss_intra_s,
            "loss_inter_s": loss_inter_s,
            "loss_t": loss_t,
            "loss_s": loss_s,
            "loss": loss,
            "pred_dT": pred_dT,
            "gt_dT": gt_dT,
        }

    def _get_N(self, crop_size: int) -> tuple[int, int]:
        N = (crop_size // self.patch_size) ** 2
        N_vis = int(N * (1 - self.mask_ratio))
        N_pos = int(N_vis * (1 - self.pos_mask_ratio))
        return N_vis, N_vis - N_pos

    def _encode_group(self, x, params, indices, seg_embed, base_offset, N_vis):
        device = x[0].device
        group_x = torch.stack([x[i] for i in indices], dim=1)
        group_params = torch.stack([params[i] for i in indices], dim=1)
        enc = self.encode_views(group_x)
        enc["z_enc"] = enc["z_enc"] + seg_embed[indices].unsqueeze(0).unsqueeze(2)

        B = group_x.shape[0]
        cls_tokens = enc["z_enc"][:, :, 0, :]
        flat_patches = enc["z_enc"][:, :, 1:, :].flatten(1, 2)
        flat_positions = enc["patch_positions_vis"].flatten(1, 2)

        M = enc["ids_remove_pos"].shape[2]
        view_offsets = (torch.arange(len(indices), device=device) * N_vis).view(
            1, len(indices), 1
        )
        flat_ids_remove = (enc["ids_remove_pos"] + view_offsets + base_offset).view(
            B, -1
        )

        expanded_params = (
            group_params[:, :, 4:8].unsqueeze(2).expand(-1, -1, M, -1).flatten(1, 2)
        )

        shape = (group_x.shape[-2], M, len(indices))
        return (
            cls_tokens,
            flat_patches,
            flat_positions,
            flat_ids_remove,
            expanded_params,
            shape,
        )

    def forward(self, x: list[Tensor], params: list[Tensor]) -> dict:
        device = x[0].device
        V = len(x)
        assert V <= self.num_views

        # Group views by their resolution
        resolutions = [inp.shape[-1] for inp in x]
        groups = [
            list(group) for _, group in groupby(range(V), key=lambda i: resolutions[i])
        ]

        # Precompute per-view visible‐token counts and base offsets
        N_vis_list = [self._get_N(res)[0] for res in resolutions]
        view_base_offset = [0] + list(accumulate(N_vis_list[:-1]))

        # Apply segment embeddings
        seg_embed = self.segment_embed[:V]
        if self.segment_embed_mode == "permute":
            seg_embed = seg_embed[torch.randperm(V, device=device)]

        # Encode each group via our helper, then unpack
        results = [
            self._encode_group(
                x, params, g, seg_embed, view_base_offset[g[0]], N_vis_list[g[0]]
            )
            for g in groups
        ]
        # Unpack grouped results and concatenate tensors
        *tensor_groups, shapes_list = zip(*results)
        joint_cls, joint_patches, joint_positions, joint_ids_remove, joint_params = [
            torch.cat(group, dim=1) for group in tensor_groups
        ]
        joint_latents = torch.cat([joint_cls, joint_patches], dim=1)

        # Decode, drop masked tokens, compute loss
        patch_pred = self.forward_decoder(joint_latents)["pose_pred"][
            :, joint_cls.shape[1] :, :
        ]
        pose_pred_nopos, patch_pos_nopos = self._drop_pos(
            patch_pred, joint_positions, joint_ids_remove
        )
        loss_dict = self._forward_loss(
            pose_pred_nopos, patch_pos_nopos, joint_params, shapes_list
        )

        return {
            "patch_positions_nopos": patch_pos_nopos,
            "shapes": shapes_list,
            **loss_dict,
        }


def PART_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = PARTMaskedAutoEncoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
PART_mae_vit_base_patch16 = (
    PART_mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
)

if __name__ == "__main__":
    from src.data.components.transforms.multi_crop_v3 import ParametrizedMultiCropV3
    from PIL import Image
    import torch.utils._pytree as pytree
    from lightning import seed_everything

    seed_everything(42)
    backbone = PART_mae_vit_base_patch16(pos_mask_ratio=0.75, mask_ratio=0.75).cuda()
    gV, lV = 2, 4
    V = gV + lV
    t = ParametrizedMultiCropV3(n_global_crops=gV, n_local_crops=lV, distort_color=True)
    print(t.compute_max_scale_ratio_aug())  # <5.97

    class MockedDataset(torch.utils.data.Dataset):
        def __init__(self, transform=None, n=4):
            self.img = Image.open("artifacts/labrador.jpg")
            self.transform = transform
            self.n = n

        def __getitem__(self, idx):
            return self.transform(self.img)

        def __len__(self):
            return self.n

    dataset = MockedDataset(t)
    seed_everything(42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    seed_everything(42)
    batch = next(iter(loader))
    seed_everything(42)
    batch = pytree.tree_map_only(
        Tensor, lambda x: x.cuda(), batch
    )  # Move to GPU if available

    seed_everything(42)
    out = backbone(*batch)
    print("Output keys:", out.keys())
    print("loss", out["loss"].detach().item())
