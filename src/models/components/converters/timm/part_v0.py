import torch

def preprocess(state_dict):
    # Remove ignored keys
    args = state_dict["args"]
    state_dict = state_dict["model"] # Renamed from ckpt to state_dict for clarity within the function
    if "base_patch16_224" in args.model:
        embed_dim = 768
        num_heads = 12
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    assert not args.use_pe, "Positional embeddings are not used in this conversion"

    if not (args.use_pe):
        # This block was empty in the provided snippet, assuming it's for future use or was elided
        pass

    state_dict.pop("cls_token", None)
    state_dict.pop("pos_embed", None) # Assuming positional embeddings are not used and should be removed
    state_dict.pop("targets", None)
    state_dict.pop("clf.weight", None)
    state_dict.pop("clf.bias", None)
    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)

    # Determine number of blocks by finding the highest block number in keys
    num_blocks = 0
    if any(key.startswith("blocks.") for key in state_dict.keys()):
        num_blocks = (
            max(int(key.split(".")[1]) for key in state_dict.keys() if key.startswith("blocks."))
            + 1
        )
    print(f"Found {num_blocks} blocks to process")

    # Get necessary dimensions from args
    # These are typically consistent for DeiT base and small models
    head_dim = embed_dim // num_heads
    
    # For ViT, the input dimension to attention's linear layers is embed_dim
    in_features_for_attn = embed_dim

    # Convert q, kv weights to qkv format
    for i in range(num_blocks):
        # Q weights and biases
        q_w = state_dict.pop(f"blocks.{i}.attn.q.weight")
        q_b = state_dict.pop(f"blocks.{i}.attn.q.bias", None) # Use None if bias might be missing

        # KV weights and biases
        kv_w_orig = state_dict.pop(f"blocks.{i}.attn.kv.weight")
        kv_b_orig = state_dict.pop(f"blocks.{i}.attn.kv.bias", None) # Use None if bias might be missing

        # De-interleave KV weights
        # Original kv_w_orig shape: (embed_dim * 2, in_features_for_attn)
        # Reshaped: (num_heads, 2 * head_dim, in_features_for_attn)
        kv_w_reshaped = kv_w_orig.reshape(num_heads, 2 * head_dim, in_features_for_attn)
        
        k_weights_per_head = kv_w_reshaped[:, 0:head_dim, :]       # Shape: (num_heads, head_dim, in_features_for_attn)
        v_weights_per_head = kv_w_reshaped[:, head_dim:2*head_dim, :] # Shape: (num_heads, head_dim, in_features_for_attn)

        # Concatenate to form K_all_weights and V_all_weights
        # Target shape for k_w_corrected and v_w_corrected: (embed_dim, in_features_for_attn)
        k_w_corrected = k_weights_per_head.reshape(num_heads * head_dim, in_features_for_attn)
        v_w_corrected = v_weights_per_head.reshape(num_heads * head_dim, in_features_for_attn)

        # Combine Q, K, V weights
        state_dict[f"blocks.{i}.attn.qkv.weight"] = torch.cat([q_w, k_w_corrected, v_w_corrected], dim=0)

        if q_b is not None and kv_b_orig is not None:
            # De-interleave KV biases
            # Original kv_b_orig shape: (embed_dim * 2)
            # Reshaped: (num_heads, 2 * head_dim)
            kv_b_reshaped = kv_b_orig.reshape(num_heads, 2 * head_dim)

            k_bias_per_head = kv_b_reshaped[:, 0:head_dim]       # Shape: (num_heads, head_dim)
            v_bias_per_head = kv_b_reshaped[:, head_dim:2*head_dim] # Shape: (num_heads, head_dim)

            # Concatenate to form K_all_bias and V_all_bias
            # Target shape for k_b_corrected and v_b_corrected: (embed_dim)
            k_b_corrected = k_bias_per_head.reshape(num_heads * head_dim)
            v_b_corrected = v_bias_per_head.reshape(num_heads * head_dim)
            
            state_dict[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q_b, k_b_corrected, v_b_corrected], dim=0)
        elif q_b is not None: # Only Q bias exists, KV bias was False
             # This case implies qkv_bias was true for q, but false for kv, which is unusual.
             # Timm's qkv layer usually has one bias for all three or none.
             # If only q_b exists, we might need to pad k and v biases with zeros
             # or ensure timm model is created with qkv_bias=False if no biases are truly present.
             # For simplicity, if kv_b_orig is None, we assume no bias for qkv.
             # If your src model had q_bias=True, kv_bias=False, this conversion will be problematic
             # for a standard timm qkv layer that expects a single bias term.
             # The safest is to ensure qkv_bias matches in both models.
             # If q_b exists but kv_b_orig doesn't, and timm expects a full qkv_bias, this will error.
             # Let's assume if any bias is present, all are, or none are.
             # If q_b is not None and kv_b_orig is None, this indicates a mismatch in bias configuration.
             # For now, if kv_b_orig is None, we don't create qkv.bias, assuming qkv_bias=False in target.
             pass
        
        # If q_b is None and kv_b_orig is None, no qkv.bias is created, which is correct if qkv_bias=False.

    return state_dict



if __name__ == "__main__":
    import timm
    from timm.models.vision_transformer import VisionTransformer
    from src.models.components.part_v0 import VisionTransformer as PARTVisionTransformer

    ckpt = torch.load(
        "artifacts/tasks_2/9u72ktsg6k/artifacts/checkpoint_epoch_200.pth",
        weights_only=False,
    )
    args = ckpt["args"]
    if ckpt["args"].model == "deit_base_patch16_224":
        from src.models.components.part_v0 import partvit_base_patch16_224

        model_cls = partvit_base_patch16_224
    elif ckpt["args"].model == "deit_small_patch16_224":
        from src.models.components.part_v0 import partvit_small_patch16_224

        model_cls = partvit_small_patch16_224
    else:
        raise ValueError(f"Unknown model type: {ckpt['args'].model}")

    src = model_cls(
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        mask_prob=args.mask_prob,
        # pretrain=args.pretrain,
        pretrain=False,
        linear_probe=args.linear_probe,
        global_pool=args.global_pool,
        use_pe=args.use_pe,
        use_ce=args.use_ce,
        with_replacement=args.with_replacement,
        num_pairs=args.num_pairs,
        ce_op=args.ce_op,
        loss=args.loss,
        head_type=args.head_type,
        column_embedding=args.column_embedding,
        num_channels=2,
        debug_pairwise_mlp=args.debug_pairwise_mlp,
        debug_cross_attention=args.debug_cross_attention,
        cross_attention_query_type=args.cross_attention_query_type,
    ).eval()
    src.load_state_dict(ckpt["model"], strict=False)
    # assert src.pretrain, "Pretrained model expected"

    ckpt = preprocess(ckpt)

    # load it to deit_base_patch16_224
    tgt = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=0,
        pos_embed="none",
        class_token=False,
        global_pool="avg",
        fc_norm=False,
        pretrained_strict=True,
        pretrained_cfg_overlay=dict(
            state_dict=ckpt
        )
    ).eval()
    # miss, unex = tgt.load_state_dict(ckpt, strict=True)
    # assert len(miss) == 0, "There are missing keys in the state_dict"
    # assert len(unex) == 0, "There are unexpected keys in the state_dict"
    x = torch.randn(1, 3, 224, 224)

    tgt_out = dict()
    src_out = dict()


    tgt_out["x_tok"] = tgt.patch_embed(x)
    src_out["x_tok"] = src.patch_embed(x)
    assert torch.allclose(tgt_out["x_tok"], src_out["x_tok"], atol=1e-5), (
        "Patch embeddings do not match!"
    )

    # testing block:
    z = torch.randn(1, 196, 768)  # assuming 196 patches for deit_base_patch16_224
    # src_qkv = torch.cat([src.blocks[0].attn.q(z), src.blocks[0].attn.kv(z)], dim=-1)
    # tgt_qkv = tgt.blocks[0].attn.qkv(z)
    # assert torch.allclose(src_qkv, tgt_qkv, atol=1e-5), (
    #     "QKV weights do not match!"
    # ) # attn weights match :)

    tgt_att = tgt.blocks[0].attn(z)
    src_att = src.blocks[0].attn(z, None)
    assert torch.allclose(tgt_att, src_att, atol=1e-5), (
        "Block attns output does not match!"
    ) # not match :(
    tgt_bout = tgt.blocks[0](z)
    src_bout = src.blocks[0](z, None)
    assert torch.allclose(tgt_bout, src_bout, atol=1e-5), (
        "Block output does not match!"
    )


    tgt_out["z"] = tgt.forward_features(x)
    src_out["z"] = src.forward_features_pretrain(x)
    assert torch.allclose(tgt_out["z"], src_out["z"], atol=1e-5), (
        "Feature extraction does not match!"
    )
