from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
from src.utils.visualization.general import paste_patch


@torch.no_grad
def reconstruction_gt(
    x: list[Float[Tensor, "C gH gW"] | Float[Tensor, "C lH lW"]],  # len(x) = V
    patch_positions_nopos: Float[Tensor, "M 2"],
    crop_params: list[Float[Tensor, "4"]],  # len(crop_params) = V
    num_tokens: list[int],  # [gM, ..., lM, lM, ..., lM] (gM #gV times, lM #lV times)
    patch_size: int,
    canonical_img_size: int,
) -> Float[Tensor, "B C canonical_img_size canonical_img_size"]:
    device = x[0].device
    C = x[0].shape[0]

    patch_positions_nopos_grouped = torch.split(patch_positions_nopos, num_tokens)

    # Initialize output canvas and count maps.
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    assert (
        len(x)
        == len(crop_params)
        == len(patch_positions_nopos_grouped)
        == len(num_tokens)
    ), (
        f"Input lists must have the same length: {len(x)}, {len(crop_params)}, {len(patch_positions_nopos_grouped)}, {len(num_tokens)}"
    )

    for crop, params, patch_positions in zip(
        x, crop_params, patch_positions_nopos_grouped
    ):
        N = patch_positions.shape[0]
        H, W = crop.shape[1:3]
        crop_size = H
        cp = params.float()  # [y, x, h, w] in canonical coordinates
        origin = cp[:2]
        size = cp[2:4]
        patch_size_canonical = patch_size * (size / crop_size)

        # For each patch position in the grid.
        for i in range(N):
            pos = patch_positions[i].float()  # in crop coordinates
            pos_canonical = origin + (pos / crop_size) * size
            paste_patch(
                crop=crop,
                pos=pos,
                pos_canonical=pos_canonical,
                patch_size_canonical=patch_size_canonical,
                canvas=canvas,
                count_map=count_map,
                patch_size=patch_size,
                canonical_size=canonical_img_size,
            )

    count_map[count_map == 0] = 1
    return canvas / count_map


@torch.no_grad
def reconstruction_vec(
    x: list[Float[Tensor, "C gH gW"] | Float[Tensor, "C lH lW"]],
    patch_positions_nopos: Float[Tensor, "M 2"],
    crop_params: list[Float[Tensor, "4"]],
    num_tokens: list[int],
    patch_size: int,
    canonical_img_size: int,
) -> Float[Tensor, "C canonical_img_size canonical_img_size"]:
    """
    Even more vectorized reconstruction: extracts & resizes patches per view,
    then scatters them all at once into the canvas and count_map.
    """
    device = patch_positions_nopos.device
    V = len(x)
    assert len(x) == len(crop_params) == len(num_tokens), (
        f"Length mismatch: {len(x)}, {len(crop_params)}, {len(num_tokens)}"
    )

    # --- Precompute per-patch canonical positions & sizes ---
    params = torch.stack([p.float() for p in crop_params], dim=0)  # [V,4]
    origins = params[:, :2]  # [V,2]
    sizes = params[:, 2:4]  # [V,2]
    crop_sizes = torch.tensor(
        [c.shape[1] for c in x], device=device, dtype=torch.float32
    )  # [V]

    view_idx = torch.arange(V, device=device).repeat_interleave(
        torch.tensor(num_tokens, device=device)
    )  # [M]

    origin_all = origins[view_idx]  # [M,2]
    size_all = sizes[view_idx]  # [M,2]
    crop_size_all = crop_sizes[view_idx]  # [M]

    patch_size_can_all = patch_size * (size_all / crop_size_all.unsqueeze(1))  # [M,2]
    pos_norm = patch_positions_nopos.float() / crop_size_all.unsqueeze(1)  # [M,2]
    pos_canon_all = origin_all + pos_norm * size_all  # [M,2]

    # --- Prepare output tensors ---
    C = x[0].shape[0]
    Hc = Wc = canonical_img_size
    canvas = torch.zeros((C, Hc, Wc), device=device)
    count_map = torch.zeros((1, Hc, Wc), device=device)

    offset = 0
    for v, crop in enumerate(x):
        nv = num_tokens[v]
        if nv == 0:
            continue

        # --- Extract & resize all patches for view v ---
        # positions in crop space
        yx = patch_positions_nopos[offset : offset + nv].long()
        H, W = crop.shape[1], crop.shape[2]
        y_crop = yx[:, 0].clamp(0, H - patch_size)
        x_crop = yx[:, 1].clamp(0, W - patch_size)

        # batched extract: unfold -> [nv, C, patch_size, patch_size]
        ct = crop.unsqueeze(0)  # [1, C, H, W]
        patches = ct.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        patches = patches[0, :, y_crop, x_crop]

        # compute this view’s canonical patch size
        ps_can = patch_size_can_all[offset]  # [2]
        h_can = max(1, min(Hc, int(ps_can[0].round().item())))
        w_can = max(1, min(Wc, int(ps_can[1].round().item())))

        # resize all at once -> [nv, C, h_can, w_can]
        patches_resized = F.interpolate(
            patches, size=(h_can, w_can), mode="bilinear", align_corners=False
        )

        # --- Build flattened indices for scatter_add ---
        pos_can = pos_canon_all[offset : offset + nv].round().long()
        y0 = pos_can[:, 0].clamp(0, Hc - h_can)
        x0 = pos_can[:, 1].clamp(0, Wc - w_can)

        ys = torch.arange(h_can, device=device)
        xs = torch.arange(w_can, device=device)
        dy, dx = torch.meshgrid(ys, xs, indexing="ij")  # [h_can, w_can]

        # [nv, h_can, w_can]
        y_idx = y0[:, None, None] + dy[None]
        x_idx = x0[:, None, None] + dx[None]

        P = nv * h_can * w_can
        y_flat = y_idx.reshape(-1)
        x_flat = x_idx.reshape(-1)
        pos_flat = y_flat * Wc + x_flat  # [P]
        pos_idx = pos_flat.unsqueeze(0).expand(C, P)  # [C, P]

        # flatten patch values to [C, P]
        vals = patches_resized.permute(1, 0, 2, 3).reshape(C, -1)

        # scatter-add into canvas
        canvas_flat = canvas.view(C, -1)
        canvas_flat.scatter_add_(1, pos_idx, vals)

        # update count_map likewise
        count_flat = count_map.view(1, -1)
        ones = torch.ones((1, P), device=device)
        count_flat.scatter_add_(1, pos_flat.unsqueeze(0), ones)

        offset += nv

    # avoid division by zero
    count_map[count_map == 0] = 1
    return canvas / count_map
