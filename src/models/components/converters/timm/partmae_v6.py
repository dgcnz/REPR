import math

def preprocess(state_dict):
    # Reshape patch embedding weights
    state_dict = state_dict["model"]
    print(state_dict["patch_embed.proj.weight"].shape)
    print(state_dict["pos_embed"].shape)
    if "register_tokens" in state_dict:
        # force convert_dinov2
        state_dict["mask_token"] = state_dict["mask_pos_token"]

    # Filter out decoder parts
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder_")}
    
    # Remove unnecessary keys
    for key in list(state_dict.keys()):
        for to_delete in ["mask_pos_token", "segment_embed", "pose_head", "dino_head"]:
            if key.startswith(to_delete):
                state_dict.pop(key, None)
    return state_dict