import torch
import torch.nn.functional as F


def ground_truth_reconstruction_from_crops_with_crop_params(
    image_crops: torch.Tensor,      # [V, C, H, W] augmented crop images
    patch_positions: torch.Tensor,  # [V, N, 2] patch positions in crop coordinates (assumed top-left)
    patch_size: int,                # patch size in the crop (e.g., 16)
    img_size: int,                  # size of the crop (e.g., 224)
    canonical_img_size: int,        # size of the canonical canvas (e.g., 512)
    crop_params: torch.Tensor       # [V, 4] each row is [y, x, h, w] used to generate the crop from canonical space
) -> torch.Tensor:
    """
    Reconstruct a canonical image from a set of augmented crops using ground truth patch positions and the crop parameters.

    For each view v:
      - Compute vertical and horizontal scale factors from crop_params:
            scale_y = crop_params[v,2] / img_size,
            scale_x = crop_params[v,3] / img_size.
      - The new patch dimensions will be:
            new_patch_h = patch_size * scale_y,
            new_patch_w = patch_size * scale_x.
      - Each patch’s top-left coordinate (in crop space) is mapped to canonical space via:
            canonical_pos = crop_params[v][:2] + (patch_pos / img_size) * crop_params[v][2:4].

    The function extracts patches from each crop, resizes them using the exact per-dimension scaling,
    and pastes them into a canonical canvas of size [C, canonical_img_size, canonical_img_size]. Overlapping areas are averaged.

    Args:
        image_crops: Tensor of shape [V, C, H, W] with augmented crop images.
        patch_positions: Tensor of shape [V, N, 2] with patch positions (top-left) in crop coordinates.
        patch_size: The patch size in the crop (e.g., 16).
        img_size: The size of the crop (e.g., 224).
        canonical_img_size: The size of the canonical canvas (e.g., 512).
        crop_params: Tensor of shape [V, 4] where each row is [y, x, h, w] describing the crop's location
                     and size in canonical space.

    Returns:
        A reconstructed canonical image of shape [C, canonical_img_size, canonical_img_size].
    """
    V, C, H, W = image_crops.shape
    _, N, _ = patch_positions.shape
    device = image_crops.device

    # Initialize a blank canonical canvas and a count map.
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    for v in range(V):
        # For view v, retrieve the crop parameters: [y, x, h, w]
        cp = crop_params[v].float()  # [y, x, h, w]
        origin = cp[:2]  # top-left coordinate of the crop in canonical space
        size = cp[2:4]   # the crop's size in canonical space

        # Compute vertical and horizontal scale factors.
        scale_y = size[0] / img_size
        scale_x = size[1] / img_size

        new_patch_h = int(round(float(patch_size * scale_y)))
        new_patch_w = int(round(float(patch_size * scale_x)))

        for i in range(N):
            # Get patch position in crop coordinates.
            pos = patch_positions[v, i].float()  # [y, x]
            # Map to canonical coordinates using the crop parameters.
            canonical_pos = origin + (pos / img_size) * size
            y_can, x_can = int(round(canonical_pos[0].item())), int(round(canonical_pos[1].item()))
            # Ensure the patch fits within the canonical canvas.
            y_can = max(0, min(canonical_img_size - new_patch_h, y_can))
            x_can = max(0, min(canonical_img_size - new_patch_w, x_can))

            # Extract the patch from the crop.
            y_crop, x_crop = int(round(pos[0].item())), int(round(pos[1].item()))
            y_crop = max(0, min(H - patch_size, y_crop))
            x_crop = max(0, min(W - patch_size, x_crop))
            patch = image_crops[v][:, y_crop:y_crop + patch_size, x_crop:x_crop + patch_size].unsqueeze(0)  # [1, C, patch_size, patch_size]

            # Resize the patch using the exact per-dimension scaling.
            patch_resized = F.interpolate(patch, size=(new_patch_h, new_patch_w),
                                          mode='bilinear', align_corners=False).squeeze(0)  # [C, new_patch_h, new_patch_w]

            # Paste the resized patch onto the canonical canvas.
            canvas[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += patch_resized
            count_map[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += 1

    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map
    return reconstructed_img




def global_optimization_reconstruction_from_crops(
    image_crops: torch.Tensor,      # [V, C, H, W] augmented crop images
    patch_positions: torch.Tensor,  # [V, N, 2] patch positions in crop coordinates (assumed top-left)
    patch_size: int,                # patch size in the crop (e.g., 16)
    img_size: int,                  # size of the crop (e.g., 224)
    canonical_img_size: int,        # size of the canonical canvas (e.g., 512)
    crop_params: torch.Tensor,      # [V, 4] each row is [y, x, h, w] used to generate the crop from canonical space
    pred_dT: torch.Tensor,          # [V, V, N, N, 4] predicted pairwise transforms (normalized)
    max_scale_ratio: float,         # maximum scale ratio for denormalization
    num_iterations: int = 100,      # number of optimization iterations
    learning_rate: float = 0.01,    # learning rate for optimization
    translation_weight: float = 5.0, # weight for translation components vs scale components
    intra_view_weight: float = 3.0   # weight for intra-view vs inter-view predictions
) -> torch.Tensor:
    """
    Reconstruct a canonical image using global optimization to find patch positions.
    
    This method optimizes in the normalized space to avoid scale imbalances and numerical instability:
    1. Sets up variables for normalized positions and scales
    2. Computes pairwise differences in normalized space
    3. Compares with the normalized predictions from the model
    4. Only denormalizes for the final reconstruction
    
    Args:
        image_crops: Tensor of shape [V, C, H, W] with augmented crop images
        patch_positions: Tensor of shape [V, N, 2] with patch positions (top-left) in crop coordinates
        patch_size: The patch size in the crop (e.g., 16)
        img_size: The size of the crop (e.g., 224)
        canonical_img_size: The size of the canonical canvas (e.g., 512)
        crop_params: Tensor of shape [V, 4] where each row is [y, x, h, w] describing the crop's location
        pred_dT: Tensor of shape [V, V, N, N, 4] with predicted pairwise transformations (normalized)
        max_scale_ratio: Maximum scale ratio for denormalization
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        translation_weight: Weight factor for translation components vs. scale components
        intra_view_weight: Weight factor for intra-view vs inter-view predictions
        
    Returns:
        reconstructed_img: A reconstructed canonical image of shape [C, canonical_img_size, canonical_img_size]
    """
    V, C, H, W = image_crops.shape
    _, N, _ = patch_positions.shape
    device = image_crops.device
    
    # ===== STEP 1: INITIALIZE OPTIMIZATION VARIABLES =====
    
    # Initialize normalized position variables (in range [0, 1])
    # After denormalization, this maps to [0, canonical_img_size]
    pos_norm = torch.zeros((V, N, 2), device=device, requires_grad=True)
    
    # Initialize normalized scale variables (centered around a base value)
    # After denormalization, this maps to log(patch_size_in_canonical)
    scale_norm = torch.zeros((V, N, 2), device=device, requires_grad=True)
    
    # Initialize positions and scales with reasonable values
    with torch.no_grad():
        # For positions: initialize in [0.3, 0.7] to keep patches mostly visible
        base_pos = 0.5  # Center of normalized space
        pos_range = 0.2  # Variation around center
        pos_norm.copy_(torch.rand_like(pos_norm) * pos_range * 2 - pos_range + base_pos)
        
        # For scales: initialize to create reasonably sized patches
        # Using a positive base value ensures patches have reasonable size
        base_scale_norm = 0.5  # This maps to exp(0.5*log(max_scale_ratio))
        scale_range = 0.1     # Small variation in initial scale
        scale_norm.copy_(torch.rand_like(scale_norm) * scale_range * 2 - scale_range + base_scale_norm)
    
    # Create optimizer with different learning rates for positions and scales
    optimizer = torch.optim.Adam([
        {'params': pos_norm, 'lr': learning_rate},
        {'params': scale_norm, 'lr': learning_rate * 0.5}  # Lower learning rate for scale
    ])
    
    # ===== STEP 2: OPTIMIZATION LOOP =====
    
    # Pre-compute view relationship masks for efficient loss calculation
    view_mask = torch.eye(V, device=device)  # Diagonal = 1 for same view
    intra_view_mask = view_mask.view(V, V, 1, 1, 1).expand(-1, -1, N, N, 4)
    inter_view_mask = (1.0 - view_mask).view(V, V, 1, 1, 1).expand(-1, -1, N, N, 4)
    
    # Number of elements for normalization
    intra_elements = V * N * N * 2.0  # For each component (translation, scale)
    inter_elements = V * (V - 1) * N * N * 2.0
    
    log_interval = max(1, num_iterations // 10)  # Log approximately 10 times
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # --- Compute all pairwise differences between patches ---
        # For each view-pair (v_i, v_j), compute difference between each patch-pair
        
        # [V, 1, N, 1, 2] - [1, V, 1, N, 2] => [V, V, N, N, 2]
        pos_diffs = pos_norm.unsqueeze(1).unsqueeze(3) - pos_norm.unsqueeze(0).unsqueeze(2)
        
        # [V, 1, N, 1, 2] - [1, V, 1, N, 2] => [V, V, N, N, 2]
        scale_diffs = scale_norm.unsqueeze(1).unsqueeze(3) - scale_norm.unsqueeze(0).unsqueeze(2)
        
        # Concatenate to get full difference tensor [V, V, N, N, 4]
        opt_diffs = torch.cat([pos_diffs, scale_diffs], dim=-1)
        
        # --- Compute loss compared to predicted transforms ---
        
        # Compute MSE separately for translation and scale components
        loss_components = torch.nn.functional.mse_loss(
            opt_diffs, pred_dT, reduction='none'
        )
        
        # Apply masks to separate intra-view and inter-view losses
        intra_loss = (loss_components * intra_view_mask).sum() / intra_elements
        inter_loss = (loss_components * inter_view_mask).sum() / inter_elements
        
        # Split losses by translation and scale
        intra_loss_t = (loss_components[..., :2] * intra_view_mask[..., :2]).sum() / (intra_elements / 2)
        intra_loss_s = (loss_components[..., 2:] * intra_view_mask[..., 2:]).sum() / (intra_elements / 2)
        inter_loss_t = (loss_components[..., :2] * inter_view_mask[..., :2]).sum() / (inter_elements / 2)
        inter_loss_s = (loss_components[..., 2:] * inter_view_mask[..., 2:]).sum() / (inter_elements / 2)
        
        # Apply weighting between intra/inter view predictions
        loss_t = (intra_view_weight * intra_loss_t + inter_loss_t) / (intra_view_weight + 1.0)
        loss_s = (intra_view_weight * intra_loss_s + inter_loss_s) / (intra_view_weight + 1.0)
        
        # Combine translation and scale losses with weighting
        transform_loss = translation_weight * loss_t + loss_s
        
        # Add boundary constraints to keep positions within [0, 1]
        pos_boundary_loss = 10.0 * torch.mean(torch.relu(torch.abs(pos_norm - 0.5) - 0.5) ** 2)
        
        # Add regularization for scale stability: keep scales near base_scale_norm
        base_scale_norm = 0.5
        scale_reg_loss = 0.01 * torch.mean((scale_norm - base_scale_norm) ** 2)
        
        # Combine all losses
        total_loss = transform_loss + pos_boundary_loss + scale_reg_loss
        
        # Backpropagate and optimize
        total_loss.backward()
        optimizer.step()
        
        # Keep positions in valid range
        with torch.no_grad():
            pos_norm.data.clamp_(0.0, 1.0)
            scale_norm.data.clamp_(0.2, 0.8)  # Prevent extreme scales
        
        # Log progress
        if iteration % log_interval == 0 or iteration == num_iterations - 1:
            print(f"Iteration {iteration}/{num_iterations}, "
                  f"T Loss: {loss_t.item():.5f}, S Loss: {loss_s.item():.5f}, "
                  f"Total: {total_loss.item():.5f}")
    
    # ===== STEP 3: RECONSTRUCT IMAGE USING OPTIMIZED POSITIONS AND SCALES =====
    
    # Denormalize positions and scales
    positions = pos_norm.detach() * canonical_img_size  # [0,1] -> [0,canonical_img_size]
    log_scales = scale_norm.detach() * torch.log(torch.tensor(max_scale_ratio, device=device))
    
    # Initialize reconstruction canvas
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)
    
    patches_placed = 0
    total_patches = V * N
    
    # For each view and patch...
    for v in range(V):
        for p in range(N):
            # Get patch from original crop
            y_crop, x_crop = patch_positions[v, p].int()
            y_crop = max(0, min(H - patch_size, y_crop))
            x_crop = max(0, min(W - patch_size, x_crop))
            patch = image_crops[v][:, y_crop:y_crop + patch_size, x_crop:x_crop + patch_size].unsqueeze(0)
            
            # Compute patch size in canonical space
            patch_h = patch_size * torch.exp(log_scales[v, p, 0]).item()
            patch_w = patch_size * torch.exp(log_scales[v, p, 1]).item()
            
            # Ensure reasonable patch size
            new_patch_h = max(4, min(canonical_img_size // 2, int(round(patch_h))))
            new_patch_w = max(4, min(canonical_img_size // 2, int(round(patch_w))))
            
            # Get canonical position
            y_can = int(round(positions[v, p, 0].item()))
            x_can = int(round(positions[v, p, 1].item()))
            
            # Check if patch fits within canvas
            if (0 <= y_can < canonical_img_size - new_patch_h and 
                0 <= x_can < canonical_img_size - new_patch_w):
                
                # Resize patch to target size
                patch_resized = torch.nn.functional.interpolate(
                    patch, size=(new_patch_h, new_patch_w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                
                # Place patch on canvas
                canvas[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += patch_resized
                count_map[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += 1
                patches_placed += 1
    
    # Report statistics
    print(f"Placed {patches_placed}/{total_patches} patches ({patches_placed/total_patches:.1%})")
    
    # Average overlapping regions
    count_map = count_map.clamp(min=1.0)  # Avoid division by zero
    reconstructed_img = canvas / count_map
    
    return reconstructed_img, positions, log_scales


def predicted_reconstruction_from_crops_with_pred_dT(
    image_crops: torch.Tensor,      # [V, C, H, W] augmented crop images
    patch_positions: torch.Tensor,  # [V, N, 2] patch positions in crop coordinates (assumed top-left)
    patch_size: int,                # patch size in the crop (e.g., 16)
    img_size: int,                  # size of the crop (e.g., 224)
    canonical_img_size: int,        # size of the canonical canvas (e.g., 512)
    crop_params: torch.Tensor,      # [V, 4] each row is [y, x, h, w] used to generate the crop from canonical space
    pred_dT: torch.Tensor,          # [V, V, N, N, 4] predicted pairwise transforms (normalized)
    max_scale_ratio: float          # maximum scale ratio for denormalization
) -> torch.Tensor:
    """
    Reconstruct a canonical image from a set of augmented crops using predicted pairwise transformations.

    This function:
    1. Selects a reference view/patch
    2. Derives absolute canonical positions for all patches using pairwise predictions
    3. Extracts patches from crops, resizes them according to predicted scales, and places them on the canvas

    Args:
        image_crops: Tensor of shape [V, C, H, W] with augmented crop images
        patch_positions: Tensor of shape [V, N, 2] with patch positions (top-left) in crop coordinates
        patch_size: The patch size in the crop (e.g., 16)
        img_size: The size of the crop (e.g., 224)
        canonical_img_size: The size of the canonical canvas (e.g., 512)
        crop_params: Tensor of shape [V, 4] where each row is [y, x, h, w] describing the crop's location
                     and size in canonical space
        pred_dT: Tensor of shape [V, V, N, N, 4] with predicted pairwise transformations (normalized)
        max_scale_ratio: Maximum scale ratio for denormalizing the predictions

    Returns:
        A reconstructed canonical image of shape [C, canonical_img_size, canonical_img_size]
    """
    V, C, H, W = image_crops.shape
    _, N, _ = patch_positions.shape
    device = image_crops.device

    # Initialize canvas and count map for reconstruction
    canvas = torch.zeros((C, canonical_img_size, canonical_img_size), device=device)
    count_map = torch.zeros((1, canonical_img_size, canonical_img_size), device=device)

    # Step 1: Denormalize predictions
    # Translation components (first 2) are normalized by canonical_img_size
    # Scale components (last 2) are normalized by log(max_scale_ratio)
    pred_dT_denorm = pred_dT.clone()
    pred_dT_denorm[..., :2] *= canonical_img_size  # Denormalize translation
    pred_dT_denorm[..., 2:] *= torch.log(torch.tensor(max_scale_ratio, device=device))  # Denormalize scale

    # Step 2: Choose reference view (view 0) and patch (patch 0)
    ref_view = 0
    ref_patch = 0

    # Step 3: Compute absolute canonical positions for all patches
    # Initialize with placeholder values
    canonical_positions = torch.zeros((V, N, 2), device=device)
    log_patch_sizes = torch.zeros((V, N, 2), device=device)

    # First, compute the reference patch's canonical position using crop parameters
    ref_cp = crop_params[ref_view].float()  # [y, x, h, w]
    ref_origin = ref_cp[:2]  # Top-left coordinate of reference crop in canonical space
    ref_size = ref_cp[2:4]   # Size of reference crop in canonical space

    # Scale factors for reference view
    ref_scale_y = ref_size[0] / img_size
    ref_scale_x = ref_size[1] / img_size

    # Map reference patch position to canonical space
    ref_pos = patch_positions[ref_view, ref_patch].float()  # [y, x] in crop coordinates
    canonical_positions[ref_view, ref_patch] = ref_origin + (ref_pos / img_size) * ref_size

    # Log patch sizes for reference patch
    log_patch_sizes[ref_view, ref_patch, 0] = torch.log(torch.tensor(patch_size * ref_scale_y, device=device))
    log_patch_sizes[ref_view, ref_patch, 1] = torch.log(torch.tensor(patch_size * ref_scale_x, device=device))

    # Step 4: Use predicted pairwise transforms to position all other patches
    # First, compute positions for all patches in reference view
    for p in range(N):
        if p == ref_patch:
            continue
        # Use the prediction from reference patch to current patch
        delta_t = pred_dT_denorm[ref_view, ref_view, ref_patch, p, :2]
        delta_s = pred_dT_denorm[ref_view, ref_view, ref_patch, p, 2:]

        # Apply translation to get canonical position
        canonical_positions[ref_view, p] = canonical_positions[ref_view, ref_patch] + delta_t

        # Apply scale to get log patch sizes
        log_patch_sizes[ref_view, p] = log_patch_sizes[ref_view, ref_patch] + delta_s

    # Next, compute positions for all patches in other views
    for v in range(V):
        if v == ref_view:
            continue
        for p in range(N):
            # Use the prediction from reference view's reference patch to current view's patch
            delta_t = pred_dT_denorm[ref_view, v, ref_patch, p, :2]
            delta_s = pred_dT_denorm[ref_view, v, ref_patch, p, 2:]

            # Apply translation to get canonical position
            canonical_positions[v, p] = canonical_positions[ref_view, ref_patch] + delta_t

            # Apply scale to get log patch sizes
            log_patch_sizes[v, p] = log_patch_sizes[ref_view, ref_patch] + delta_s

    # Step 5: Extract patches from each crop, resize and place them on the canvas
    for v in range(V):
        for p in range(N):
            # Get patch position in crop coordinates
            y_crop, x_crop = patch_positions[v, p].int()
            y_crop = max(0, min(H - patch_size, y_crop))
            x_crop = max(0, min(W - patch_size, x_crop))

            # Extract patch from crop
            patch = image_crops[v][:, y_crop:y_crop + patch_size, x_crop:x_crop + patch_size].unsqueeze(0)

            # Get target patch size in canonical space
            patch_h = torch.exp(log_patch_sizes[v, p, 0]).item()
            patch_w = torch.exp(log_patch_sizes[v, p, 1]).item()
            new_patch_h = int(round(patch_h))
            new_patch_w = int(round(patch_w))

            # Ensure minimum patch size
            new_patch_h = max(1, new_patch_h)
            new_patch_w = max(1, new_patch_w)

            # Get canonical position (top-left corner)
            y_can, x_can = canonical_positions[v, p].int()

            # Ensure the patch fits within the canvas
            if (y_can >= 0 and y_can + new_patch_h <= canonical_img_size and
                x_can >= 0 and x_can + new_patch_w <= canonical_img_size):

                # Resize patch to target size
                if new_patch_h > 0 and new_patch_w > 0:
                    patch_resized = F.interpolate(
                        patch, size=(new_patch_h, new_patch_w),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)

                    # Place patch on canvas
                    canvas[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += patch_resized
                    count_map[:, y_can:y_can + new_patch_h, x_can:x_can + new_patch_w] += 1

    # Average overlapping regions
    count_map[count_map == 0] = 1
    reconstructed_img = canvas / count_map

    return reconstructed_img