import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gradio as gr
from gradio_patch_selection import PatchSelector
import torchvision.transforms as T
from omegaconf import OmegaConf
import hydra

# Import the PART MAE ViT model
from src.models.components.partmae_v6 import PART_mae_vit_base_patch16

# Configuration
PATCH_SIZE = 16  # must match model.patch_size
IMG_SIZE = 224  # must match model.img_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing: resize to IMG_SIZE, convert to tensor, normalize with ImageNet stats
transform = T.Compose(
    [
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def load_model():
    V = 2
    RUN_ID = "nw6nhpa2"
    EPOCH = "0199"
    cfg = OmegaConf.load(f"artifacts/model-{RUN_ID}:v0/config.yaml")
    ckpt = torch.load(f"artifacts/model-{RUN_ID}:v0/epoch_{EPOCH}.ckpt")["model"]
    ckpt["pose_head.linear.weight"] = ckpt.pop("decoder_pred.weight")
    ckpt["_patch_loss.sigma"] = torch.tensor([0.1, 0.1, 0.1, 0.1])
    ckpt["segment_embed"] = ckpt["segment_embed"][:V]
    # ckpt["segment_embed"] = torch.zeros_like(ckpt["segment_embed"])[:V]

    cfg["model"]["segment_embed_mode"] = (
        "permute" if cfg["model"].pop("permute_segment_embed") else "fixed"
    )
    model = hydra.utils.instantiate(
        cfg["model"],
        _target_="src.models.components.partmae_v6.PARTMaskedAutoEncoderViT",
        verbose=True,
        num_views=V,
    )
    model.load_state_dict(ckpt, strict=True)
    return model


# Initialize the model: create default, then configure sampler and masking
# model = PART_mae_vit_base_patch16(num_views=2).to(DEVICE)
model = load_model().to(DEVICE).eval()
# Use update_conf instead of constructor overrides
model.update_conf(sampler="ongrid_canonical", mask_ratio=0.0, pos_mask_ratio=1.0)


# Helper: compute full pred_dT for all patch pairs (A<->A, A<->B, etc.)
def compute_pred_dT_full(pilA: Image.Image, pilB: Image.Image):
    # Preprocess and batch
    tA = transform(pilA).unsqueeze(0).to(DEVICE)
    tB = transform(pilB).unsqueeze(0).to(DEVICE)
    x = torch.stack([tA, tB], dim=1)

    with torch.no_grad():
        enc = model.encode_views(x)
        z_enc = enc["z_enc"]  # [1, 2, 1+N, D]
        ids_remove = enc["ids_remove_pos"]

        B, V, NP1, D = z_enc.shape
        N = NP1 - 1
        cls_tokens = z_enc[:, :, 0, :].reshape(B, V, D)
        patch_tokens = z_enc[:, :, 1:, :].reshape(B, V * N, D)

        joint_z = torch.cat([cls_tokens.reshape(B, V, D), patch_tokens], dim=1)
        z_dec = model.forward_decoder(joint_z)

        joint_ids_remove = ids_remove.view(B, -1)
        pred_dT = model.pose_head(z_dec[:, V:, :], joint_ids_remove)
        pred_dT = torch.clamp(pred_dT, -1.0, 1.0)

    return pred_dT.squeeze(0).cpu().numpy(), N


def compute_pred_dT_full_v2(pilA: Image.Image, pilB: Image.Image):
    # Preprocess and batch
    tA = transform(pilA).unsqueeze(0).to(DEVICE)
    tB = transform(pilB).unsqueeze(0).to(DEVICE)
    x = [tA, tB]
    params = [torch.randn(1, 8, device=DEVICE), torch.randn(1, 8, device=DEVICE)]

    with torch.no_grad():
        out = model(x, params)
        pred_dT = out["pred_dT"]  # [1, 2, N, 4]
        pred_dT = torch.clamp(pred_dT, -1.0, 1.0)
        pred_dT = pred_dT.squeeze(0)
        T = pred_dT.shape[0]
        ids = out["joint_ids_remove"][0]
        ids = torch.argsort(ids)
        rows = ids.unsqueeze(1).expand(T, T)
        cols = ids.unsqueeze(0).expand(T, T)
        pred_dT = pred_dT[rows, cols]
    return pred_dT.cpu().numpy()


# Helper: overlay distance heatmap on image B
def overlay_distance_heatmap(
    pilB: Image.Image, pred_dT_full: np.ndarray, N: int, patch_idx_A: int
):
    start_B = N
    dists = np.linalg.norm(
        pred_dT_full[patch_idx_A, start_B : start_B + N, :2], axis=-1
    )

    grid_size = int(np.sqrt(N))
    heat = dists.reshape(grid_size, grid_size)
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    cmap = plt.get_cmap("jet")
    heat_rgba = (cmap(heat_norm) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert("RGBA")
    heat_img = heat_img.resize(pilB.size, resample=Image.BILINEAR)

    base = pilB.convert("RGBA")
    overlay = Image.blend(base, heat_img, alpha=0.5)
    return overlay.convert("RGB")


# Gradio callback to update overlay
# Accepts PatchSelector annotation dict and Image B (could be numpy or PIL)
def update_overlay(annotation: dict, imageB):
    # annotation: {'image': array or PIL, 'patchIndex': int}
    if annotation is None or imageB is None:
        return None, "Please upload both images and select a patch."

    imgA = annotation.get("image", None)
    idx = annotation.get("patchIndex", None)
    if imgA is None or idx is None:
        return None, "No patch selected."

    # Convert annotation image to PIL if needed
    if isinstance(imgA, np.ndarray):
        pilA = Image.fromarray(imgA)
    elif isinstance(imgA, Image.Image):
        pilA = imgA
    else:
        pilA = Image.open(imgA)

    # Convert imageB to PIL if needed
    if isinstance(imageB, np.ndarray):
        pilB = Image.fromarray(imageB)
    elif isinstance(imageB, Image.Image):
        pilB = imageB
    else:
        pilB = Image.open(imageB)

    pred_dT_full = compute_pred_dT_full_v2(pilA, pilB)
    assert pred_dT_full.shape[0] % 2 == 0, "pred_dT_full should be even"
    N = pred_dT_full.shape[0] // 2
    overlay = overlay_distance_heatmap(pilB, pred_dT_full, N, idx)
    info = f"Patch A index: {idx} → grid {int(np.sqrt(N))}×{int(np.sqrt(N))}"
    return overlay, info


# Build Gradio UI
demo = gr.Blocks()
with demo:
    gr.Markdown(
        """
        ## Patch Translation Heatmap Visualizer
        Upload an image and see predicted patch translations.
        Click on a patch in the grid to visualize distances on Image B.
        """
    )

    with gr.Row():
        with gr.Column():
            patch_selector = PatchSelector(
                {"image": Image.open("artifacts/samoyed.jpg"), "patchIndex": 0},
                img_size=IMG_SIZE,
                patch_size=PATCH_SIZE,
                show_grid=True,
                grid_color="rgba(200,200,200,0.5)",
            )
            info_box = gr.Textbox(label="Info", interactive=False)

        with gr.Column():
            imageB_input = gr.Image(
                type="pil",
                label="Image B",
                value="artifacts/samoyed_rot.jpg",
            )
            overlay_output = gr.Image(type="pil", label="Heatmap Overlay")

    # Allow uploading a new image into the PatchSelector
    patch_selector.upload(
        fn=lambda img: {"image": img, "patchIndex": 0},
        inputs=[patch_selector],
        outputs=[patch_selector],
    )

    # Update heatmap when a patch is selected
    patch_selector.patch_select(
        fn=update_overlay,
        inputs=[patch_selector, imageB_input],
        outputs=[overlay_output, info_box],
    )

if __name__ == "__main__":
    demo.launch()


# if __name__ == "__main__":
#
#     pilA, pilB = Image.open("artifacts/samoyed.jpg"), Image.open("artifacts/samoyed.jpg")
#     idx = 60 #
#     pred_dT_full = compute_pred_dT_full_v2(pilA, pilB)
#     print(pred_dT_full.shape)
#     assert pred_dT_full.shape[0] % 2 == 0, "pred_dT_full should be even"
#     N = pred_dT_full.shape[0] // 2
#     overlay = overlay_distance_heatmap(pilB, pred_dT_full, N, idx)
#     info = f"Patch A index: {idx} → grid {int(np.sqrt(N))}×{int(np.sqrt(N))}"
#     print(info)
#     overlay.show()
#     # return overlay, info
