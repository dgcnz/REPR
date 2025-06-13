import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gradio as gr
# Import the PART MAE ViT model
from gradio_patch_selection import PatchSelector
import torch
import torchvision.transforms as T
from jaxtyping import Float
from omegaconf import OmegaConf
import hydra
# Configuration
PATCH_SIZE = 16  # must match model.patch_size
IMG_SIZE = 224  # must match model.img_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing: convert to tensor, normalize with ImageNet stats
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def load_model() -> torch.nn.Module:
    """Load the pretrained PART model for inference."""
    V = 2
    RUN_ID = "nw6nhpa2"
    EPOCH = "0199"
    cfg = OmegaConf.load(f"artifacts/model-{RUN_ID}:v0/config.yaml")
    ckpt = torch.load(f"artifacts/model-{RUN_ID}:v0/epoch_{EPOCH}.ckpt")["model"]
    ckpt["pose_head.linear.weight"] = ckpt.pop("decoder_pred.weight")
    ckpt["_patch_loss.sigma"] = torch.tensor([0.1, 0.1, 0.1, 0.1])
    ckpt["segment_embed"] = torch.zeros_like(ckpt["segment_embed"])[:V]

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


# Initialize the model and configure sampler
model = load_model().to(DEVICE).eval()
model.update_conf(sampler="ongrid_canonical", mask_ratio=0.0, pos_mask_ratio=1.0)


def compute_pred_logvar_full(
    pilA: Image.Image, pilB: Image.Image
) -> Float[np.ndarray, "T T 4"]:
    """Return predicted log-variance for all patch pairs.

    :param pilA: First image resized to ``IMG_SIZE``.
    :param pilB: Second image resized to ``IMG_SIZE``.
    :returns: Array of shape ``[T, T, 4]`` with log-variance values.
    """
    tA = transform(pilA).unsqueeze(0).to(DEVICE)
    tB = transform(pilB).unsqueeze(0).to(DEVICE)
    x = [tA, tB]
    params = [torch.randn(1, 8, device=DEVICE), torch.randn(1, 8, device=DEVICE)]

    with torch.no_grad():
        out = model(x, params)
        logvar = out["var_dT"]
        if logvar is None:
            raise RuntimeError("Model did not return var_dT")
        logvar = logvar.squeeze(0)
        T = logvar.shape[0]
        ids = out["joint_ids_remove"][0]
        ids = torch.argsort(ids)
        rows = ids.unsqueeze(1).expand(T, T)
        cols = ids.unsqueeze(0).expand(T, T)
        logvar = logvar[rows, cols]
    return logvar.cpu().numpy()


def overlay_uncertainty_heatmap(
    pilB: Image.Image,
    logvar_full: Float[np.ndarray, "T T 4"],
    N: int,
    patch_idx_A: int,
) -> Image.Image:
    """Overlay uncertainty heatmap for a selected patch.

    :param pilB: Base image B at ``IMG_SIZE``.
    :param logvar_full: Log-variance array ``[T, T, 4]``.
    :param N: Number of patches per image.
    :param patch_idx_A: Index of the selected patch in image A.
    :returns: Image with overlay visualizing uncertainty.
    """
    start_B = N
    var = np.exp(logvar_full[patch_idx_A, start_B : start_B + N, :2]).mean(axis=-1)
    grid_size = int(np.sqrt(N))
    heat = var.reshape(grid_size, grid_size)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    cmap = plt.get_cmap("jet")
    heat_rgba = (cmap(heat) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert("RGBA")
    heat_img = heat_img.resize(pilB.size, resample=Image.BILINEAR)

    base = pilB.convert("RGBA")
    overlay = Image.blend(base, heat_img, alpha=0.5)
    return overlay.convert("RGB")


def update_overlay(annotation: dict, imageB) -> tuple[Image.Image | None, str]:
    """Gradio callback to compute and overlay uncertainties.

    :param annotation: Output of :class:`PatchSelector` with selected patch.
    :param imageB: Second image (numpy array or PIL Image).
    :returns: Tuple of overlay image and information string.
    """
    if annotation is None or imageB is None:
        return None, "Please upload both images and select a patch."

    imgA_data = annotation.get("image")
    idx = annotation.get("patchIndex")
    if imgA_data is None or idx is None:
        return None, "No patch selected."

    if isinstance(imgA_data, np.ndarray):
        pilA = Image.fromarray(imgA_data)
    elif isinstance(imgA_data, Image.Image):
        pilA = imgA_data
    else:
        pilA = Image.open(imgA_data)

    if isinstance(imageB, np.ndarray):
        pilB = Image.fromarray(imageB)
    elif isinstance(imageB, Image.Image):
        pilB = imageB
    else:
        pilB = Image.open(imageB)

    pilA = pilA.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    pilB = pilB.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    logvar_full = compute_pred_logvar_full(pilA, pilB)
    assert logvar_full.shape[0] % 2 == 0, "logvar_full should be even"
    N = logvar_full.shape[0] // 2

    overlay = overlay_uncertainty_heatmap(pilB, logvar_full, N, idx)
    info = f"Patch A index: {idx} → grid {int(np.sqrt(N))}×{int(np.sqrt(N))}"
    return overlay, info


def resize_img(img: Image.Image) -> Image.Image:
    """Resize uploaded image to ``IMG_SIZE``."""
    return img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


# Build Gradio UI
demo = gr.Blocks()
with demo:
    gr.Markdown(
        """
        ## Patch Uncertainty Visualizer
        Upload two images and select a patch on Image A to view the predicted
        uncertainty over Image B.
        """
    )

    with gr.Row():
        with gr.Column():
            patch_selector = PatchSelector(
                {
                    "image": Image.open("artifacts/dog.webp").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                    "patchIndex": 0,
                },
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
                value=Image.open("artifacts/human-right.jpg").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
            )
            overlay_output = gr.Image(type="pil", label="Heatmap Overlay")

    imageB_input.upload(fn=resize_img, inputs=imageB_input, outputs=imageB_input)

    patch_selector.upload(
        fn=lambda img: {"image": img, "patchIndex": 0},
        inputs=[patch_selector],
        outputs=[patch_selector],
    )

    patch_selector.patch_select(
        fn=update_overlay,
        inputs=[patch_selector, imageB_input],
        outputs=[overlay_output, info_box],
    )

if __name__ == "__main__":
    demo.launch()
