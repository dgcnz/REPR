import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import gradio as gr
from gradio_patch_selection import PatchSelector
import torchvision.transforms as T
from omegaconf import OmegaConf
import hydra
import logging
from torch.utils._pytree import tree_map_only

logging.basicConfig(level=logging.INFO)

# Configuration
PATCH_SIZE = 16  # must match model.patch_size
IMG_SIZE = 224  # must match model.img_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing: convert to tensor, normalize with ImageNet stats
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def load_model(run_folder: Path, ckpt_name: str) -> torch.nn.Module:
    """Load the pretrained PART model for inference."""
    cfg = OmegaConf.load(run_folder / ".hydra" / "config.yaml")
    ckpt = torch.load(run_folder / ckpt_name)["model"]
    model = hydra.utils.instantiate(
        cfg["model"],
        _target_="src.models.components.partmae_v6.PARTMaskedAutoEncoderViT",
        num_views=2,  # we're using 2 views for this example
        sampler="ongrid_canonical",
        mask_ratio=0.0,
        pos_mask_ratio=1.0,
    )
    model.load_state_dict(ckpt, strict=True)
    return model


RUN_FOLDER = Path("outputs/2025-06-13/19-56-13")
CKPT_NAME = "last.ckpt"
# Initialize the model and configure sampler
model = load_model(RUN_FOLDER, CKPT_NAME).to(DEVICE).eval()


def compute_pred_dT_full_v2(pilA: Image.Image, pilB: Image.Image):
    # Preprocess and batch
    tA = transform(pilA).to(DEVICE)
    tB = transform(pilB).to(DEVICE)
    x = [torch.stack([tA, tB]).unsqueeze(0)]  # Stack images into a batch
    params = [torch.randn(1, 2, 8, device=DEVICE)]

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
    # sharpen distances using softmax
    dists = -dists

    grid_size = int(np.sqrt(N))
    # revert distances so that 1 is close, 0 is far
    heat = dists.reshape(grid_size, grid_size)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    cmap = plt.get_cmap("jet")
    heat_rgba = (cmap(heat) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert("RGBA")
    heat_img = heat_img.resize(pilB.size, resample=Image.BILINEAR)

    base = pilB.convert("RGBA")
    overlay = Image.blend(base, heat_img, alpha=0.5)
    return overlay.convert("RGB")


# Gradio callback to update overlay
# Accepts PatchSelector annotation dict and Image B (could be numpy or PIL)
def update_overlay(annotation: dict, imageB):
    # annotation: {'image': array or PIL, 'patchIndex': int}
    logging.info(
        f"Received annotation : {tree_map_only(np.ndarray, lambda x: x.shape, annotation)}"
    )
    if annotation is None or imageB is None:
        return None, "Please upload both images and select a patch."

    imgA_data = annotation.get("image", None)
    idx = annotation.get("patch_index", None)
    if imgA_data is None or idx is None:
        return None, "No patch selected."

    # Convert annotation image (imgA_data) to PIL.
    # imgA_data from PatchSelector is expected to be a PIL Image at IMG_SIZE.
    if isinstance(imgA_data, np.ndarray):
        pilA = Image.fromarray(imgA_data)
    elif isinstance(imgA_data, Image.Image):
        pilA = imgA_data
    else:
        pilA = Image.open(imgA_data)

    # Convert imageB to PIL if needed
    if isinstance(imageB, np.ndarray):
        pilB = Image.fromarray(imageB)
    elif isinstance(imageB, Image.Image):
        pilB = imageB
    else:
        pilB = Image.open(imageB)  # pilB is at original resolution

    # Resize pilB to IMG_SIZE. This is used for model input and overlay.
    pilA = pilA.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    pilB = pilB.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    pred_dT_full = compute_pred_dT_full_v2(pilA, pilB)
    assert pred_dT_full.shape[0] % 2 == 0, "pred_dT_full should be even"
    N = pred_dT_full.shape[0] // 2

    # Use the resized pilB (pilB) for the overlay function
    overlay = overlay_distance_heatmap(pilB, pred_dT_full, N, idx)
    info = f"Patch A index: {idx} → grid {int(np.sqrt(N))}×{int(np.sqrt(N))}"
    return overlay, info


def resize_img(img):
    # Resize the image to the desired size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img


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
                {
                    "image": Image.open("artifacts/samoyed.jpg").resize(
                        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
                    ),
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
                value=Image.open("artifacts/dog.jpg").resize(
                    (IMG_SIZE, IMG_SIZE), Image.LANCZOS
                ),
                # width=IMG_SIZE,
                # height=IMG_SIZE,
            )
            overlay_output = gr.Image(type="pil", label="Heatmap Overlay")

    imageB_input.upload(fn=resize_img, inputs=imageB_input, outputs=imageB_input)

    # Allow uploading a new image into the PatchSelector
    patch_selector.upload(
        # fn=lambda img: {"image": img, "patchIndex": 0},
        fn=lambda inputs: inputs,
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
