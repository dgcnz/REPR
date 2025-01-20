import pytest
from PIL import Image
import torchvision.transforms.v2.functional as TTFv2
import hydra
from omegaconf import OmegaConf
import torch
from src.models.part_vit_module import PARTViTModule
import lightning as L
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def get_img():
    img = Image.open("artifacts/img.jpg")
    img = img.resize((224, 224))
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    return img, patch_size, num_patches


def get_cfg():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="train.yaml", overrides=["experiment=overfit_part_im1k_pairdiff_mlp", "model/criterion_fn=l1", "model.sample_mode=offgrid"])
    OmegaConf.register_new_resolver("eval", eval)
    return cfg



def get_train_transform():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="train.yaml", overrides=["experiment=overfit_part_im1k_pairdiff_mlp", "model/criterion_fn=l1", "model.sample_mode=ongrid"])

    transform = hydra.utils.instantiate(cfg.data.train_transform)
    return transform






if __name__ == "__main__":
    # let's test the loss function on initialization

    cfg = get_cfg()   
    module: PARTViTModule = hydra.utils.instantiate(cfg.model)
    transform = hydra.utils.instantiate(cfg.data.train_transform)
    x_raw, _, _ = get_img()

    x = transform(x_raw).unsqueeze(0)
    # Function to run experiment and plot results
    # Store input features
    proj_input_features = []
    proj_output_features = []
    backbone_features = []
    
    def hook_fn(module, input, output):
        proj_input_features.append(input[0].detach().cpu()) # [B, NP, D] , D=768
        proj_output_features.append(output.detach().cpu())

    def backbone_hook_fn(module, input, output):
        backbone_features.append(output.detach().cpu())
    
    def run_and_plot(init_type, fig_row, module):
        # Clear stored features
        proj_input_features.clear()
        proj_output_features.clear()
        backbone_features.clear()
        
        # Register hooks
        proj_hook = module.net.proj.register_forward_hook(hook_fn)
        backbone_hook = module.net.backbone.register_forward_hook(backbone_hook_fn)

        import math
        if init_type == "normal":
            fan_in = module.net.proj.weight.shape[1]
            print(fan_in)
            std = 1.0 / (5.0 * math.sqrt(fan_in))
            module.net.proj.weight.data.normal_(0, std)
            module.net.proj.bias.data.fill_(0)
        elif init_type == "uniform":
            fan_in = module.net.proj.weight.shape[1]
            # bound = math.sqrt(3.0/(25.0 * fan_in))
            bound = 1.0/(5.0 * math.sqrt(3.0 * fan_in))
            module.net.proj.weight.data.uniform_(-bound, bound)
            module.net.proj.bias.data.fill_(0)
        else: # default
            pass
            
        out = module.model_step({"image": x})
        
        # Remove hooks
        proj_hook.remove()
        backbone_hook.remove()

        # Separate x and y components for selected tensors
        gt_T_y, gt_T_x = module.norm_T(out["gt_T"].detach().cpu())[0].unbind(dim=-1)
        pred_T_y, pred_T_x = module.norm_T(out["pred_T"].detach().cpu())[0].unbind(dim=-1)
        weights_y, weights_x = module.net.proj.weight.detach().cpu()
        proj_out_y, proj_out_x = proj_output_features[0][0].unbind(dim=-1)

        # Define the data mapping
        data = {
            'proj_output': (proj_out_x, proj_out_y, 2, 3, -1, 1),
            'gt_T': (gt_T_x, gt_T_y, 4, 5, -1, 1),
            'pred_T': (pred_T_x, pred_T_y, 6, 7, -1, 1),
            'weights': (weights_x, weights_y, 8, 9, None, None),
        }
        # plot proj_input
        fig.add_trace(
            go.Histogram(
                x=proj_input_features[0].flatten(),
                nbinsx=100,
                name=f"{init_type}_proj_input"
            ),
            row=fig_row,
            col=1
        )

        # Add traces for each data type
        for name, (x_data, y_data, x_col, y_col, mn, mx) in data.items():
            # Add x component histogram
            fig.add_trace(
            go.Histogram(
                x=x_data.flatten(),
                nbinsx=100,
                name=f"{init_type}_{name}_x",
                # add minimum and maximum values
                xbins=dict(start=mn, end=mx) if mn is not None else None
            ),
            row=fig_row,
            col=x_col
            )

            # Add y component histogram
            fig.add_trace(
            go.Histogram(
                x=y_data.flatten(),
                nbinsx=100,
                name=f"{init_type}_{name}_y",
                xbins=dict(start=mn, end=mx) if mn is not None else None
            ),
            row=fig_row,
            col=y_col
            )

        return out["loss"]
        # Create figure with subplots
    fig = make_subplots(rows=3, cols=9, 
                        subplot_titles=["proj_input", "proj_output_x", "proj_output_y", "gt_T_x", "gt_T_y", "pred_T_x", "pred_T_y", "weights_x", "weights_y"] * 3,  
                        row_titles=["Normal Init", "Uniform Init", "Default Init"])

    # Run experiments
    from copy import deepcopy
    normal_loss = run_and_plot("normal", 1, deepcopy(module))
    uniform_loss = run_and_plot("uniform", 2, deepcopy(module))
    default_loss = run_and_plot("default", 3, deepcopy(module))

    fig.update_layout(
        title_text=f"Comparison of Normal vs Uniform Initialization\nNormal Loss: {normal_loss:.4f}, Uniform Loss: {uniform_loss:.4f}, Default Loss: {default_loss:.4f}",
        showlegend=True,
        height=2000,
        width=2000
    )
    
    print("showing plot")
    fig.show()
