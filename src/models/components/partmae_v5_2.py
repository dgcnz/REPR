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


        # Build token crop sizes: for each group, create a (M*V, 1) tensor filled with the crop size.
        token_crop_sizes = torch.cat(
            [torch.full((M * V, 1), cs, device=device) for cs, M, V in shapes]
        )[None].expand(B, -1, -1)

        # # Directly cast zip(*shapes) to a tensor.
        # _, Ms, Vs = list(zip(*shapes))
        # # Total unique views and per-view repetition factors.
        # repeats_list = list(chain.from_iterable([[m] * v for m, v in zip(Ms, Vs)]))
        # label_list = list(
        #     chain.from_iterable(([i] * r for i, r in enumerate(repeats_list)))
        # )

        label_list = []
        label = 0
        for cs, M, V in shapes:
            # For each view in the current shape, repeat the current label M times.
            for _ in range(V):
                label_list.extend([label] * M)
                label += 1


        # Convert the computed list to a tensor on the correct device, adding a new dimension as needed.
        labels = torch.tensor(label_list, device=device)[None]

        # Compute pairwise differences.
        pred_dT = self._compute_pred(pose_pred_nopos)
        gt_dT = self._compute_gt(patch_pos_nopos, params, token_crop_sizes)
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        # Build intra- and inter-view masks.
        mask = (labels.unsqueeze(2) == labels.unsqueeze(1)).float().unsqueeze(-1)
        mask = mask.expand(B, -1, -1, 1)
        diag, offdiag = mask.sum(), (1 - mask).sum()

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

    def forward(self, x, params) -> dict:
        # Ensure inputs are lists.
        x = x if isinstance(x, list) else [x]
        device = x[0].device
        params = params if isinstance(params, list) else [params]

        perm = torch.randperm(self.num_views, device=device)
        seg_embed_all = (
            self.segment_embed[perm]
            if self.segment_embed_mode == "permute"
            else self.segment_embed
        )

        # Determine group counts based on resolution (assumes x is ordered by resolution).
        resolutions = [inp.shape[-1] for inp in x]
        counts = [len(list(group)) for _, group in groupby(resolutions)]
        end_idx = list(accumulate(counts))
        start_idx = [0] + list(end_idx)[:-1]
        N_vis = [self._get_N(res)[0] for res in resolutions]
        base_offset = list(accumulate([N_vis[i] * c for i, c in enumerate(counts)]))
        base_offset = [0] + list(accumulate(base_offset))[:-1]

        # Initialize accumulators and offsets.
        nV = len(counts)
        all_cls, all_patches, all_positions = _list(nV), _list(nV), _list(nV)
        all_ids_remove, all_group_params, shapes_list = _list(nV), _list(nV), _list(nV)

        # Fuse group building and processing in one loop.
        for ix, count in enumerate(counts):
            indices = list(range(start_idx[ix], end_idx[ix]))
            group_x = torch.stack([x[i] for i in indices], dim=1)  # [B, k, C, H, W]
            group_params = torch.stack([params[i] for i in indices], dim=1)  # [B, k, 8]
            enc = self.encode_views(group_x)
            enc["z_enc"] += (
                seg_embed_all[start_idx[ix] : end_idx[ix]].unsqueeze(0).unsqueeze(2)
            )
            B = group_x.shape[0]
            cls_tokens = enc["z_enc"][:, :, 0, :]
            patch_tokens = enc["z_enc"][:, :, 1:, :]
            flat_patches = patch_tokens.flatten(1, 2)
            flat_positions = enc["patch_positions_vis"].flatten(1, 2)
            M = enc["ids_remove_pos"].shape[2]
            view_offsets = (
                torch.arange(count, device=device) * N_vis[ix]
            ).view(1, count, 1)
            flat_ids_remove = (
                enc["ids_remove_pos"] + view_offsets + base_offset[ix]
            ).view(B, -1)
            expanded_params = (
                group_params[:, :, 4:8].unsqueeze(2).expand(-1, -1, M, -1).flatten(1, 2)
            )
            shapes_list[ix] = (group_x.shape[-2], M, count)
            all_cls[ix] = cls_tokens
            all_patches[ix] = flat_patches
            all_positions[ix] = flat_positions
            all_ids_remove[ix] = flat_ids_remove
            all_group_params[ix] = expanded_params

        joint_cls = torch.cat(all_cls, dim=1)
        joint_patches = torch.cat(all_patches, dim=1)
        joint_positions = torch.cat(all_positions, dim=1)
        joint_ids_remove = torch.cat(all_ids_remove, dim=1)
        joint_params = torch.cat(all_group_params, dim=1)
        joint_latents = torch.cat([joint_cls, joint_patches], dim=1)

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

    backbone = PART_mae_vit_base_patch16(pos_mask_ratio=0.75, mask_ratio=0.75).cuda()
    gV, lV = 2, 4
    V = gV + lV
    t = ParametrizedMultiCropV3(n_global_crops=gV, n_local_crops=lV, distort_color=True)
    print(t.compute_max_scale_ratio_aug())  # <5.97

    class MockedDataset(torch.utils.data.Dataset):
        def __init__(self, transform=None):
            self.img = Image.open("artifacts/labrador.jpg")
            self.transform = transform

        def __getitem__(self, idx):
            return self.transform(self.img)

        def __len__(self):
            return 4

    dataset = MockedDataset(t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    batch = pytree.tree_map_only(
        Tensor, lambda x: x.cuda(), batch
    )  # Move to GPU if available
    
    out = backbone(*batch)
    print("Output keys:", out.keys())
