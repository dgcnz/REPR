import torch
from torch import nn, Tensor
from functools import partial
from torch.nn.modules.utils import _pair
from src.models.components import partmae_v5


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
        shapes: list,  # each element is a tuple: (crop_size, M, k)
    ) -> dict:
        B, T, _ = pose_pred_nopos.shape
        device = pose_pred_nopos.device

        # Build crop sizes and view labels per group.
        crop_sizes_list = []
        labels_list = []
        label_counter = 0
        for crop_size, M, k in shapes:
            group_crop_sizes = torch.full((k * M, 1), crop_size, device=device)
            crop_sizes_list.append(group_crop_sizes)
            for _ in range(k):
                labels_list.append(torch.full((M,), label_counter, device=device))
                label_counter += 1

        token_crop_sizes = torch.cat(crop_sizes_list, dim=0).unsqueeze(0).expand(B, -1, -1)
        labels = torch.cat(labels_list, dim=0)  # [total_tokens]

        pred_dT = self._compute_pred(pose_pred_nopos)
        gt_dT = self._compute_gt(patch_pos_nopos, params, token_crop_sizes)
        loss_full = self.criterion(pred_dT, gt_dT, reduction="none")

        # Build pairwise label masks.
        labels_matrix = labels.unsqueeze(0)
        intra_mask = (
            (labels_matrix.unsqueeze(2) == labels_matrix.unsqueeze(1))
            .float()
            .unsqueeze(-1)
            .expand(B, -1, -1, 1)
        )
        inter_mask = 1 - intra_mask

        loss_t_full = loss_full[..., :2]
        loss_s_full = loss_full[..., 2:]
        diag_denom = intra_mask.sum()
        offdiag_denom = inter_mask.sum()
        loss_intra_t = (loss_t_full * intra_mask).sum() / diag_denom
        loss_inter_t = (loss_t_full * inter_mask).sum() / offdiag_denom
        loss_intra_s = (loss_s_full * intra_mask).sum() / diag_denom
        loss_inter_s = (loss_s_full * inter_mask).sum() / offdiag_denom
        loss_t = loss_inter_t * self.alpha_t + loss_intra_t * (1 - self.alpha_t)
        loss_s = loss_inter_s * self.alpha_s + loss_intra_s * (1 - self.alpha_s)
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
        N_nopos = N_vis - N_pos
        return N_vis, N_nopos

    def _get_group_info(self, x: list) -> list:
        """
        Group inputs by resolution and collect necessary metadata for processing.
        Returns a list of dictionaries with metadata for each group.
        """
        # Identify groups by resolution
        resolutions = [inp.shape[-1] for inp in x]
        res_tensor = torch.tensor(resolutions)
        _, counts = torch.unique_consecutive(res_tensor, return_counts=True)
        
        # Build group info
        groups = []
        start_idx, seg_offset, base_offset = 0, 0, 0
        
        for count in counts:
            end_idx = start_idx + count.item()
            indices = list(range(start_idx, end_idx))
            crop_size = x[indices[0]].shape[-2]
            N_vis, _ = self._get_N(crop_size)
            
            groups.append({
                "indices": indices,
                "crop_size": crop_size,
                "k": len(indices),
                "N_vis": N_vis,
                "seg_offset": seg_offset,
                "base_offset": base_offset
            })
            
            start_idx = end_idx
            seg_offset += len(indices)
            base_offset += len(indices) * N_vis
            
        return groups

    def forward(self, x, params) -> dict:
        # Ensure inputs are lists
        if not isinstance(x, list):
            x = [x]
        if not isinstance(params, list):
            params = [params]

        # Get segment embeddings (with optional permutation)
        seg_embed_all = (
            self.segment_embed[torch.randperm(self.num_views)]
            if self.permute_segment_embed
            else self.segment_embed
        )
        
        # Group inputs by resolution
        group_infos = self._get_group_info(x)
        
        # Process all groups and collect results
        all_cls, all_patches, all_positions = [], [], []
        all_ids_remove, all_group_params, shapes_list = [], [], []
        
        for info in group_infos:
            indices = info["indices"]
            k = info["k"]
            N_vis = info["N_vis"]
            seg_offset = info["seg_offset"]
            base_offset = info["base_offset"]
            
            # Stack crops and parameters for current group
            group_x = torch.stack([x[i] for i in indices], dim=1)  # [B, k, C, H, W]
            group_params = torch.stack([params[i] for i in indices], dim=1)  # [B, k, 8]
            
            # Encode the group
            enc = self.encode_views(group_x)
            
            # Add segment embeddings
            seg_slice = seg_embed_all[seg_offset:seg_offset + k]  # [k, D]
            enc["z_enc"] = enc["z_enc"] + seg_slice.unsqueeze(0).unsqueeze(2)
            
            # Extract tokens
            B = group_x.shape[0]
            cls_tokens = enc["z_enc"][:, :, 0, :]  # [B, k, D]
            patch_tokens = enc["z_enc"][:, :, 1:, :]  # [B, k, N, D]
            
            # Flatten patch dimension
            flat_patches = patch_tokens.flatten(1, 2)  # [B, k*N, D]
            flat_positions = enc["patch_positions_vis"].flatten(1, 2)  # [B, k*N, 2]
            
            # Adjust masked indices with offset
            M = enc["ids_remove_pos"].shape[2]
            view_offsets = (torch.arange(k, device=enc["ids_remove_pos"].device) * N_vis).view(1, k, 1)
            adjusted_ids = enc["ids_remove_pos"] + view_offsets + base_offset
            flat_ids_remove = adjusted_ids.view(B, -1)
            
            # Expand parameters for loss computation
            group_params_crop = group_params[:, :, 4:8]  # [B, k, 4]
            expanded_params = group_params_crop.unsqueeze(2).expand(-1, -1, M, -1).flatten(1, 2)
            
            # Save shape info
            shapes_list.append((group_x.shape[-2], M, k))
            
            # Collect outputs
            all_cls.append(cls_tokens)
            all_patches.append(flat_patches)
            all_positions.append(flat_positions)
            all_ids_remove.append(flat_ids_remove)
            all_group_params.append(expanded_params)
        
        # Concatenate all outputs
        joint_cls = torch.cat(all_cls, dim=1)
        joint_patches = torch.cat(all_patches, dim=1)
        joint_positions = torch.cat(all_positions, dim=1)
        joint_ids_remove = torch.cat(all_ids_remove, dim=1)
        joint_params = torch.cat(all_group_params, dim=1)
        
        # Combine class and patch tokens
        joint_latents = torch.cat([joint_cls, joint_patches], dim=1)
        
        # Decode and compute loss
        dec_out = self.forward_decoder(joint_latents)
        total_cls = joint_cls.shape[1]
        patch_pred = dec_out["pose_pred"][:, total_cls:, :]
        
        # Drop positions for loss computation
        pose_pred_nopos, patch_pos_nopos = self._drop_pos(
            patch_pred, joint_positions, joint_ids_remove
        )
        
        # Compute loss
        loss_dict = self._forward_loss(
            pose_pred_nopos, patch_pos_nopos, joint_params, shapes_list
        )
        
        # Prepare output
        out = {"patch_positions_nopos": patch_pos_nopos}
        out.update(loss_dict)
        return out


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
    backbone = PART_mae_vit_base_patch16(pos_mask_ratio=0.75, mask_ratio=0.75)
    # x = torch.randn(1, 3, 224, 224).repeat(2, 1, 1, 1)
    # params = torch.tensor([[0, 0, 224, 224], [0, 0, 224, 224]])
    # patch_size = 16
    # num_patches = (224 // patch_size) ** 2
    # out = backbone.forward(x, params)
    # print("Output keys:", out.keys())
    # print(out["loss"])

    # Test forward pass
    B, gV, lV = 4, 2, 6
    g_x = torch.randn(B, gV, 3, 224, 224)
    g_params = torch.randn(B, gV, 8)
    l_x = torch.randn(B, lV, 3, 96, 96)
    l_params = torch.randn(B, lV, 8)
    out = backbone(g_x, g_params, l_x, l_params)
    print("Output keys:", out.keys())
