# NOTE: this script must be run on hummingbird's conda env  
# conda run --no-capture-output  -n hummingbird python -m scripts.run_hbird_eval
import timm
import torch
import re
from hbird.hbird_eval import hbird_evaluation # type: ignore


def process_droppos(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def process_simdino(state_dict):
    return {
        k.replace("_orig_mod.", "").replace("backbone.", ""): v
        for k, v in state_dict["teacher"].items()
    }


def process_simdinov2(ckpt):
    ckpt = ckpt["teacher"]
    ckpt = {
        k.removeprefix("backbone."): v
        for k, v in ckpt.items()
        if k.startswith("backbone")
    }
    # revert chunk weight
    ckpt = {
        re.sub(r"blocks\.(\d+)\.(\d+)\.", r"blocks.\2.", k): v for k, v in ckpt.items()
    }
    return ckpt



def preprocess_part(ckpt):
    # Remove ignored keys
    ckpt = ckpt["model"]
    ignored_keys = ["targets", "clf.weight", "clf.bias", "head.weight", "head.bias"]
    for key in ignored_keys:
        ckpt.pop(key, None)

    # zero-out pos embeds
    ckpt["pos_embed"] = torch.zeros_like(ckpt["pos_embed"])

    # Determine number of blocks by finding the highest block number in keys
    num_blocks = (
        max(int(key.split(".")[1]) for key in ckpt.keys() if key.startswith("blocks."))
        + 1
    )
    print(f"Found {num_blocks} blocks to process")

    # Convert q, kv weights to qkv format
    for i in range(num_blocks):
        block_prefix = f"blocks.{i}.attn"

        # Get the weights
        q_weight = ckpt.pop(f"{block_prefix}.q.weight")
        kv_weight = ckpt.pop(f"{block_prefix}.kv.weight")
        q_bias = ckpt.pop(f"{block_prefix}.q.bias")
        kv_bias = ckpt.pop(f"{block_prefix}.kv.bias")

        # Combine weights and biases
        qkv_weight = torch.cat([q_weight, kv_weight], dim=0)
        qkv_bias = torch.cat([q_bias, kv_bias])

        # Store new combined weights
        ckpt[f"{block_prefix}.qkv.weight"] = qkv_weight
        ckpt[f"{block_prefix}.qkv.bias"] = qkv_bias

    return ckpt


models = {
    "untrained": lambda: dict(model_name="vit_base_patch16_224", pretrained=False),
    "dino": lambda: dict(model_name="vit_base_patch16_224.dino", pretrained=True),
    "mae": lambda: dict(model_name="vit_base_patch16_224.mae", pretrained=True),
    "neco_dino": lambda: dict(
        model_name="vit_base_patch16_224",
        pretrained=True,
        pretrained_cfg_overlay=dict(
            hf_hub_id="FunAILab/NeCo",
            hf_hub_filename="vit-base/dino-architectures/neco_on_dino_vit16_teacher.ckpt",
        ),
        pretrained_strict=False,
    ),
    "partmae_v5_2": lambda: dict(
        model_name="vit_base_patch16_224",
        pretrained=True,
        pretrained_cfg_overlay=dict(
            hf_hub_id="dgcnz/PART",
            hf_hub_filename="model-nw6nhpa2:v0/backbone_0199.ckpt",
        ),
        pretrained_strict=False,
    ),
    "partmae_v6": lambda: dict(
        model_name="vit_base_patch16_224",
        pretrained=True,
        num_classes=0,
        pretrained_cfg_overlay=dict(
            state_dict=process_partmaev6(
                torch.load(
                    "../PART/outputs/2025-05-27/20-14-09/last.ckpt", # morning-breeze-236, 42r7tltv
                    # "../PART/outputs/2025-05-27/17-28-24/last.ckpt",  # giddy-serenity-232, o72prp1r
                    # "../PART/outputs/2025-05-28/11-34-14/last.ckpt",  # 
                )
            ),
        ),
        pretrained_strict=True,
    ),
    "droppos": lambda: dict(
        model_name="vit_base_patch16_224",
        num_classes=0,
        pretrained=True,
        pretrained_strict=True,
        pretrained_cfg_overlay=dict(
            state_dict=process_droppos(
                torch.load(
                    "../PART/artifacts/DropPos_pretrain_vit_base_patch16.pth",
                )
            ),
        ),
    ),
    "simdino": lambda: dict(
        model_name="vit_base_patch16_224",
        num_classes=0,
        pretrained=True,
        pretrained_strict=False,
        pretrained_cfg_overlay=dict(
            state_dict=process_simdino(
                torch.load(
                    "../PART/artifacts/vitb16_SimDINOv1_gpu8_bs64_ep100.pth",
                )
            ),
        ),
    ),
    "simdinov2": lambda: dict(
        model_name="vit_base_patch14_reg4_dinov2",
        patch_size=16,
        img_size=224,
        num_classes=0,
        pretrained=True,
        pretrained_strict=False,
        pretrained_cfg_overlay=dict(
            state_dict=process_simdinov2(
                torch.load("../PART/artifacts/vitb16_reg4_SimDNIOv2_ep100.pth")
            )
        ),
    ),
    "part": lambda: dict(
        # model_name="deit_base_patch16_224",
        model_name="deit_base_patch16_224",
        num_classes=0,
        pos_embed="learn",  # checkpoint will overwrite this to zeros
        pretrained=True,
        pretrained_strict=False,
        pretrained_cfg_overlay=dict(
            state_dict=preprocess_part(
                torch.load(
                    "../PART/artifacts/tasks_2/9u72ktsg6k/artifacts/checkpoint_epoch_200.pth",
                    weights_only=False,
                )
            ),
        ),
    ),
}

# only keep simdino
models = {k: v for k, v in models.items() if k in ["partmae_v6"]}


device = "cuda"
# batch_size = 64
batch_size = 128 # fits on 12gb vram 4060ti
# Dataset Configurations
split = "voc"
dataset_name = "voc"
data_dir = f"/mnt/sdb1/datasets/{split}/VOCSegmentation"
train_fs_path = f"/mnt/sdb1/datasets/{split}/VOCSegmentation/sets/trainaug.txt"
val_fs_path = f"/mnt/sdb1/datasets/{split}/VOCSegmentation/sets/val.txt"


def extract_timm_features(model, imgs):
    # Extract the features from the model
    with torch.no_grad():
        features = model.forward_features(imgs)
    # Remove the first token (CLS token)
    return features[:, model.num_prefix_tokens:], None


results = dict()
for model_name, model_cfg in models.items():
    model = timm.create_model(**model_cfg()).eval()
    embed_dim = model.embed_dim
    patch_size = model.patch_embed.proj.weight.shape[-1]
    input_size = model.patch_embed.img_size[-1]
    print(
        f"Model: {model_name}, Input size: {input_size}, Patch size: {patch_size}, Embed dim: {embed_dim}"
    )

    hbird_miou = hbird_evaluation(
        model.to(device),
        d_model=embed_dim,  # size of the embedding feature vectors of patches
        patch_size=patch_size,
        batch_size=batch_size,
        input_size=input_size,
        augmentation_epoch=1,  # how many iterations of augmentations to use on top of the training dataset in order to generate the memory
        device=device,
        return_knn_details=False,  # whether to return additional NNs details
        nn_method="faiss",
        n_neighbours=30,  # the number of neighbors to fetch per image patch
        nn_params=None,  # Other parameters to be used for the k-NN operator
        ftr_extr_fn=extract_timm_features,  # function that extracts features from a vision encoder on images
        dataset_name=dataset_name,  # the name of the dataset to use, currently only Pascal VOC is included.
        data_dir=data_dir,  # path to the dataset to use for evaluation
        memory_size=None,
        train_fs_path=train_fs_path,
        val_fs_path=val_fs_path,
    )
    # print('Hummingbird Evaluation (mIoU):', hbird_miou)
    print(f"Model: {model_name}, mIoU: {hbird_miou}")
    results[model_name] = hbird_miou


# print results
for model_name, mIoU in results.items():
    print(f"Model: {model_name}, mIoU: {mIoU}")
