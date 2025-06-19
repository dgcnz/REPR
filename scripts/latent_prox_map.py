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
import timm
import logging
from src.models.components.converters.timm.partmae_v6 import preprocess
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
    ckpt_path = run_folder / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=0,
    )
    # OPTION 1: MASK POS EMBED
    # cls_pos_embed = state_dict["pos_embed"][:, :1, :]  # [1, 1, D]
    # pos_embed = state_dict["pos_embed"][:, 1:, :]  # [1, N, D]
    # pos_embed = state_dict["mask_pos_token"].expand_as(pos_embed)
    # state_dict["pos_embed"] = torch.cat([cls_pos_embed, pos_embed], dim=1)  # [1, N+1, D]
    # OPTION 2: ZERO POS EMBED
    # state_dict["pos_embed"] = torch.zeros_like(state_dict["pos_embed"])
    state_dict = preprocess(ckpt)
    model.load_state_dict(state_dict, strict=True)
    return model

CFG_FOLDER = Path("fabric_configs/experiment/hummingbird/model")
MODEL = "partmae_v6_s"
CKPT_PATH = Path("outputs/2025-06-17/10-52-57/last.ckpt")
model_conf = OmegaConf.load(CFG_FOLDER / f"{MODEL}.yaml")
model_conf.ckpt_path = CKPT_PATH
model = hydra.utils.instantiate(model_conf, _convert_='all')["net"].eval().cuda()
# Initialize the model and configure sampler
# model = load_model(RUN_FOLDER, CKPT_NAME).to(DEVICE).eval()


def compute_sim(pilA: Image.Image, pilB: Image.Image):
    # Preprocess and batch
    xA = transform(pilA).to(DEVICE)
    xB = transform(pilB).to(DEVICE)
    x = torch.stack([xA, xB], dim=0)  # [2, C, H, W]

    with torch.no_grad():
        feats = model.forward_features(x)
        # l2 normalize features
        # remove cls token
        feats = feats[:, model.num_prefix_tokens :, :]  # [B, N, D]
        feats = torch.nn.functional.normalize(feats, dim=-1) 
        # B N D

    # we're interested in seeing the distance (cosine) between patches in images A and B
    # not inside the same image.
    # so the final similarity matrix should be: N N 
    # sim[i, j] = similarity between patch i in image A and patch j in image B
    featA, featB = feats[0], feats[1]  # each is [N, D]
    sim = featA @ featB.t()  
    print(sim.shape)
    return sim.cpu().numpy()  # convert to numpy for further processing


# Helper: overlay distance heatmap on image B
def overlay_distance_heatmap(
    pilB: Image.Image, sim: np.ndarray, N: int, patch_idx_A: int
):
    grid_size = int(np.sqrt(N))
    sim = sim[patch_idx_A] # get distances from patch A to all patches in B
    # revert distances so that 1 is close, 0 is far
    heat = sim.reshape(grid_size, grid_size)
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

    sim = compute_sim(pilA, pilB)
    N = sim.shape[0]

    overlay = overlay_distance_heatmap(pilB, sim, N, idx)
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
                value=Image.open("artifacts/samoyed.jpg").resize(
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

