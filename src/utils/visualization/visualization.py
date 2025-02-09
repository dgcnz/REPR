import torch
from torch import Tensor
from jaxtyping import Float, Int
from typing import Any
from collections import defaultdict
from PIL import Image
import torchvision.transforms.v2.functional as TTFv2
import networkx as nx
import colorstamps
import matplotlib.pyplot as plt


def patch_id_to_position(patch_id: int, patch_size: int, img_size: int):
    # only works for ongrid and square images
    return (
        (patch_id // (img_size // patch_size)) * patch_size,
        (patch_id % (img_size // patch_size)) * patch_size,
    )


def get_patch_pair_gt_transform(
    patch_src_id: int, patch_tgt_id: int, patch_positions: Int[Tensor, "n_patches 2"]
):
    patch_src_y, patch_src_x = patch_positions[patch_src_id]
    patch_tgt_y, patch_tgt_x = patch_positions[patch_tgt_id]
    return torch.tensor([patch_tgt_y - patch_src_y, patch_tgt_x - patch_src_x])


def compute_reconstruction_graph(
    patch_pair_indices: Int[Tensor, "n_pairs 2"],
    patch_positions: Int[Tensor, "n_patches 2"],
) -> nx.Graph:
    """
    Return a graph where each node represents a patch and each edge's weight represents the predicted transformation.

    If add_inverse_edges is True, for each edge u->v (T=t), add an edge v->u (T=-t).

    :param patch_pair_indices: (n_pairs, 2) tensor of patch pair indices
    """
    edges = []
    for pair_idx in range(patch_pair_indices.size(0)):
        patch_idx, patch_jdx = patch_pair_indices[pair_idx]
        gt_T = get_patch_pair_gt_transform(patch_idx, patch_jdx, patch_positions)
        gt_l1 = gt_T.abs().sum().item()
        edges.append((patch_idx.item(), patch_jdx.item(), {"gt_l1": gt_l1}))

    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def traverse(
    g: dict[int, list],
    node: int,
    visited: dict[int, bool],  # global mutable
    cur_t: Float[Tensor, "2"],  # local immutable
    transforms: dict,  # global immutable
    # min_node: int, # local immutable
) -> int:
    """
    Compute the transformation of all the patches with respect to the current patch.
    Return the number of patches in the connected component.
    """
    visited[node] = True
    num = 1
    for neighbor, t in g[node]:
        if not visited[neighbor]:
            new_t = cur_t + t
            transforms[neighbor] = new_t
            num += traverse(g, neighbor, visited, new_t, transforms)
    return num


def iterative_traverse(
    g: dict[int, list],
    root: int,
    visited: dict[int, bool] = dict(),  # global mutable
) -> int:
    """
    Compute the transformation of all the patches with respect to the current patch.
    Return the number of patches in the connected component.
    """
    if visited[root]:
        return 0

    visited[root] = True
    num = 1
    transforms = {root: torch.zeros(2)}
    stack = [(root, torch.zeros(2))]
    min_node = root
    min_dist = torch.zeros(2)
    while stack:
        node, cur_t = stack.pop()
        for neighbor, t in g[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                num += 1
                new_t = cur_t + t
                transforms[neighbor] = new_t
                # l1 norm min
                if torch.norm(new_t, p=1) < torch.norm(min_dist, p=1):
                    min_dist = new_t
                    min_node = neighbor
                stack.append((neighbor, new_t))
    return num, min_node, transforms


def compute_connected_components(g: dict[int, list]):
    visited = defaultdict(bool)
    components = []
    for node in g:
        if not visited[node]:
            sz, min_node, transforms = iterative_traverse(g, node, visited)
            components.append((sz, min_node, node, transforms))
    return components


def nx_iterative_traverse(
    g: nx.MultiDiGraph,
    root: int,
    visited: dict[int, bool],  # global mutable
) -> int:
    """
    Compute the transformation of all the patches with respect to the current patch.
    Return the number of patches in the connected component.
    """
    if visited[root]:
        return 0, None, None

    visited[root] = True
    num = 1
    transforms = {root: torch.zeros(2)}
    min_node, min_dist = root, torch.zeros(2)
    # important, low distance first because it large distances are more likely to be wrong
    stack = [(root, torch.zeros(2))]
    nodes = set({int(root)})
    while stack:
        u, t_ru = stack.pop()
        for v, uv_attr in sorted(
            g[u].items(), key=lambda e: torch.norm(e[1]["T"], p=1), reverse=True
        ):
            if not visited[v]:
                visited[v] = True
                nodes.add(v)
                num += 1
                t_rv = t_ru + uv_attr["T"]
                transforms[v] = t_rv
                if torch.norm(t_rv, p=1) < torch.norm(min_dist, p=1):
                    min_dist = t_rv
                    min_node = v
                stack.append((v, t_rv))
            elif v != root and v in nodes:
                # if we have already visited the node, we can check if the transform is the same
                # if not, we have made an error, we should average the transforms
                t_rv = t_ru + uv_attr["T"]
                # average the transforms
                # transforms[v] = (transforms[v] + t_rv) / 2
                if torch.norm(t_rv, p=1) > torch.norm(transforms[v], p=1):
                    transforms[v] = t_rv
    return nodes, min_node, transforms


def nx_compute_connected_components(g: nx.MultiDiGraph):
    ## use dfs_edges, sort by l1 norm of transform/weight
    components = []
    visited = defaultdict(bool)
    for possible_root in g:
        if not visited[possible_root]:
            nodes, min_node, transforms = nx_iterative_traverse(
                g, possible_root, visited
            )
            components.append((nodes, min_node, possible_root, transforms))
    return components


import matplotlib.pyplot as plt


def plot_transform_distribution(
    transforms: Float[Tensor, "n_pairs 2"],
    name: str,
    axes: tuple[plt.Axes, plt.Axes],
):
    """
    Plot boxplot of the transform distribution.
    """
    dx = transforms[:, 0]
    dy = transforms[:, 1]
    axes[0].boxplot(dy)
    axes[0].set_title(f"{name} dy")
    axes[1].boxplot(dx)
    axes[1].set_title(f"{name} dx")


def reconstruct_image(
    transform: Float[Tensor, "n_pairs 2"],
    pair_indices: Int[Tensor, "n_pairs 2"],
    patch_positions: Int[Tensor, "n_patches 2"],
    patch_size: int,
    img: Float[Tensor, "3 H W"],
) -> Float[Tensor, "3 H W"]:
    """
    Reconstruct the image using the predicted transformations.
    """
    n_pairs, _ = transform.size()
    n_patches, _ = patch_positions.size()
    assert tuple(transform.shape) == (n_pairs, 2)
    assert tuple(pair_indices.shape) == (n_pairs, 2)
    assert tuple(patch_positions.shape) == (n_patches, 2)
    assert img.size(0) == 3
    assert img.dim() == 3

    g = compute_reconstruction_graph(pair_indices, transform, patch_positions)
    components = nx_compute_connected_components(g)

    if len(components) > 1:
        print(
            "The graph is not connected, only plotting the largest connected component"
        )

    largest_component = max(components)
    nodes, min_node, root, transforms = largest_component
    root = min(nodes, key=lambda x: torch.norm(patch_positions[x].float(), p=1))
    # compute the min_node
    # re-root the tree at the node with the smallest l1 norm (the top-leftmost node)
    _, _, transforms = nx_iterative_traverse(g, root, defaultdict(bool))
    assert transforms is not None

    new_img = torch.zeros_like(img)
    _, H, W = img.shape
    for node, t in transforms.items():
        y, x = patch_positions[node]
        # TODO: use some interpolation here?
        t_rounded = t.round().to(torch.int64)
        new_y, new_x = patch_positions[root] + t_rounded
        # TODO: validate that indices fit within the image
        if 0 <= new_y <= H - patch_size and 0 <= new_x <= W - patch_size:
            new_img[:, new_y : new_y + patch_size, new_x : new_x + patch_size] = img[
                :, y : y + patch_size, x : x + patch_size
            ]

    return new_img


def get_transforms_from_reference_patch_batch(
    refpatch_ids: list[int],
    transforms: Float[Tensor, "B n_pairs 2"],
    pair_indices: Int[Tensor, "B n_pairs 2"],
    patch_positions: Int[Tensor, "B n_patches 2"],
):
    return [
        get_transforms_from_reference_patch(
            refpatch_ids[i], transforms[i], pair_indices[i], patch_positions[i]
        )
        for i in range(len(refpatch_ids))
    ]


def get_transforms_from_reference_patch(
    refpatch_id: int,
    transforms: Float[Tensor, "n_pairs 2"],
    pair_indices: Int[Tensor, "n_pairs 2"],
    patch_positions: Int[Tensor, "n_patches 2"],
) -> dict[int, Tensor]:
    """
    Get the transforms of all patches with respect to the reference patch.

    :param refpatch_id: index of the reference patch wrt to patch_positions
    :param transforms: (n_pairs, 2) tensor of predicted transforms
    :param pair_indices: (n_pairs, 2) tensor of patch pair indices
    """
    g = compute_reconstruction_graph(pair_indices, patch_positions)
    g = nx.minimum_spanning_tree(g, weight="gt_l1")
    ref_transforms = {refpatch_id: torch.zeros(2)}
    pair_transforms = {
        (u, v): T for (u, v), T in zip(pair_indices.tolist(), transforms)
    }

    for u, v in nx.dfs_edges(g, source=refpatch_id):
        T_uv = (
            pair_transforms[(u, v)]
            if (u, v) in pair_transforms
            else -pair_transforms[(v, u)]
        )
        if u == refpatch_id:
            ref_transforms[v] = T_uv
        else:
            ref_transforms[v] = ref_transforms[u] + T_uv
    return ref_transforms


def create_provenance_grid(
    patch_positions: Int[Tensor, "n_patches 2"],
    root: int,
    transforms: dict[int, Float[Tensor, "2"]],
    patch_size: int,
    img_shape: tuple[int, int],
) -> Float[Tensor, "H W 2"]:
    """
    Creates a position grid by mapping patch positions through their transforms.

    :param patch_positions: (n_patches, 2) tensor of patch positions
    :param root: index of the root patch
    :param transforms: dictionary of transforms
    :param output: empty tensor to store the output grid

    :return: filled position grid
    """
    device = patch_positions.device

    # Convert dictionary to tensors
    nodes = torch.tensor(list(transforms.keys()), device=device)
    ts = torch.stack(list(transforms.values()))

    # Get source and target positions
    source_pos = patch_positions[nodes]
    target_pos = patch_positions[root] + ts

    # Create offset grids once
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(patch_size, device=device),
            torch.arange(patch_size, device=device),
            indexing="ij",
        )
    )

    # Broadcast operations
    y_coords = (
        offsets[0].view(1, patch_size, patch_size) + source_pos[:, 0:1, None]
    ).reshape(-1)
    x_coords = (
        offsets[1].view(1, patch_size, patch_size) + source_pos[:, 1:2, None]
    ).reshape(-1)

    new_y = (
        offsets[0].view(1, patch_size, patch_size) + target_pos[:, 0:1, None]
    ).reshape(-1)
    new_x = (
        offsets[1].view(1, patch_size, patch_size) + target_pos[:, 1:2, None]
    ).reshape(-1)

    output_abs = torch.zeros(*img_shape, 2, device=device)
    # Assign values in one operation
    output_abs[y_coords, x_coords, 0] = new_y
    output_abs[y_coords, x_coords, 1] = new_x

    output_rel = torch.zeros(*img_shape, 2, device=device)
    # Assign values in one operation
    output_rel[y_coords, x_coords, 0] = new_y - y_coords
    output_rel[y_coords, x_coords, 1] = new_x - x_coords

    return output_abs, output_rel


def plot_provenances(
    refpatch_id: int,
    ref_transforms: dict[str, Tensor],
    patch_positions: Int[Tensor, "n_patches 2"],
    img_shape: tuple[int, int],
    patch_size: int,
):
    provenance_abs, provenance_rel = create_provenance_grid(
        patch_positions, refpatch_id, ref_transforms, patch_size, img_shape
    )
    fig_y_x, ax_y_x = plt.subplots(1, 2, figsize=(10, 5))
    # plot x and y separately of abs
    ax_y_x[0].imshow(provenance_abs[:, :, 0])
    ax_y_x[0].set_title("y")
    ax_y_x[1].imshow(provenance_abs[:, :, 1])
    ax_y_x[1].set_title("x")
    fig_abs_rel, ax_abs_rel = plt.subplots(1, 2, figsize=(10, 5))
    plot_provenance(ax_abs_rel[0], provenance_rel, abs=True)  # Abs
    plot_provenance(ax_abs_rel[1], provenance_rel, abs=False)  # Rel
    return fig_y_x, fig_abs_rel


def plot_provenances_batch(
    refpatch_ids: list[int],
    ref_transforms: list[dict[int, Tensor]],
    patch_positions: Int[Tensor, "B n_patches 2"],
    img_shape: tuple[int, int],
    patch_size: int,
):
    B = len(refpatch_ids)
    fig, axes = plt.subplots(B, 4, figsize=(10, 5 * B), squeeze=False)
    for i, ax in enumerate(axes):
        provenance_abs, provenance_rel = create_provenance_grid(
            patch_positions[i],
            refpatch_ids[i],
            ref_transforms[i],
            patch_size,
            img_shape,
        )
        ax[0].imshow(provenance_abs[:, :, 0])
        ax[0].set_title("y")
        ax[1].imshow(provenance_abs[:, :, 1])
        ax[1].set_title("x")
        plot_provenance(ax[2], provenance_rel, abs=True)
        plot_provenance(ax[3], provenance_rel, abs=False)
    return fig

#     def _log_plot(
#         self,
#         fig: plt.Figure,
#         name: str,
#     ):
#         if isinstance(self.logger, TensorBoardLogger):
#             self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
#         elif isinstance(self.logger, WandbLogger):
#             wandb.log({name: wandb.Image(fig)})
#         elif self.logger.__class__.__name__ == "AimLogger":
#             from aim import Image
#             from aim.pytorch_lightning import AimLogger
# 
#             self.logger: AimLogger
#             img = Image(fig)
#             self.logger.experiment.track(
#                 img,
#                 name=name,
#                 step=self.trainer.global_step,
#                 epoch=self.trainer.current_epoch,
#             )
#         else:
#             self.cli_logger.warning(
#                 f"{type(self.logger)} is unable to log the {name} image. "
#             )
#         plt.close(fig)

#     def _log_patch_pair_indices_plot(
#         self, patch_pair_indices, num_patches: int, stage: str
#     ):
#         fig, ax = plt.subplots(1, 1)
#         plot_patch_pair_coverage(ax, patch_pair_indices, num_patches)
#         self._log_plot(fig, f"{stage}/patch_pair_indices")


# def _plot_refpatch_coverage(
#     self,
#     refpatch_id,
#     patch_positions,
#     H,
#     W,
#     patch_size,
#     ref_transforms: dict,
#     stage: str,
# ):
#     """
#     Image where 1 if patch is connected to root, 0 otherwise.
#     """
#     n_patches = (H // patch_size) * (W // patch_size)
#     occupancy = torch.zeros(H, W)
#     for patch_idx in ref_transforms.keys():
#         y = patch_positions[patch_idx, 0]
#         x = patch_positions[patch_idx, 1]
#         occupancy[y : y + patch_size, x : x + patch_size] = 1
# 
#     n_coverage = len(set(ref_transforms.keys()))
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(occupancy, cmap="binary")
#     ax.set_title(f"Coverage of {refpatch_id}: {n_coverage}/{n_patches}")
#     # add legend
#     legend_elements = [
#         plt.Rectangle(
#             (0, 0), 1, 1, facecolor="black", edgecolor="black", label="Disconnected"
#         ),
#         plt.Rectangle(
#             (0, 0), 1, 1, facecolor="white", edgecolor="black", label="Connected"
#         ),
#     ]
#     ax.legend(handles=legend_elements)
#     self._log_plot(fig, f"{stage}/root_coverage")


def plot_provenance(
    ax,
    provenance: Float[Tensor, "H W 2"],
    abs: bool = True,
):
    H, W, _ = provenance.shape
    if abs:
        min_y, min_x = 0, 0
        max_y, max_x = H, W
    else:
        min_y, max_y = -H // 2, H // 2
        min_x, max_x = -W // 2, W // 2

    rgb, stamp = colorstamps.apply_stamp(
        provenance[:, :, 0],
        provenance[:, :, 1],
        "peak",
        vmin_0=min_y,
        vmax_0=max_y,
        vmin_1=min_x,
        vmax_1=max_x,
    )
    ax.imshow(rgb, origin="lower")
    overlaid_ax = stamp.overlay_ax(ax, lower_left_corner=[0.7, 0.85], width=0.2)
    overlaid_ax.set_ylabel(r"$\phi$")
    overlaid_ax.set_xlabel(r"$\omega$")


def plot_patch_pair_transform_matrices(
    patch_pair_indices: Int[Tensor, "n_pairs 2"],
    transforms: Float[Tensor, "n_pairs 2"],
    num_patches: int,
    axes: tuple[plt.Axes, plt.Axes],
    name: str,
):
    matrix_dy = torch.zeros(num_patches, num_patches)
    matrix_dx = torch.zeros(num_patches, num_patches)
    for pair_idx, (i, j) in enumerate(patch_pair_indices):
        matrix_dy[i, j] = transforms[pair_idx, 0]
        matrix_dx[i, j] = transforms[pair_idx, 1]

    min_dy, max_dy = matrix_dy.min(), matrix_dy.max()
    min_dx, max_dx = matrix_dx.min(), matrix_dx.max()
    axes[0].imshow(matrix_dy, cmap="seismic", vmin=min_dy, vmax=max_dy)
    axes[0].set_title(f"{name} dy")
    axes[1].imshow(matrix_dx, cmap="seismic", vmin=min_dx, vmax=max_dx)
    axes[1].set_title(f"{name} dx")
    # add legend for colormap
    # legend_elements = [
    #     plt.Line2D([0], [0], color="white", marker="s", markersize=10, label="0"),
    #     plt.Line2D([0], [0], color="blue", marker="s", markersize=10, label="Negative"),
    #     plt.Line2D([0], [0], color="red", marker="s", markersize=10, label="Positive"),
    # ]
    # axes[0].legend(handles=legend_elements)


def plot_patch_pair_coverage(
    ax: plt.Axes, patch_pair_indices: Float[Tensor, "n_pairs 2"], num_patches: int
):
    ppi_matrix = torch.zeros(num_patches, num_patches)
    ppi_matrix[patch_pair_indices[:, 0], patch_pair_indices[:, 1]] = 1
    ax.imshow(ppi_matrix, cmap="binary")
    ax.set_title("Patch Pair Indices")
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="black", edgecolor="black", label="Disconnected"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="black", label="Connected"
        ),
    ]
    ax.legend(handles=legend_elements)


if __name__ == "__main__":
    img = Image.open("artifacts/img.jpg")
    img = img.resize((224, 224))
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    pair_indices = torch.stack(
        [torch.zeros(num_patches), torch.arange(num_patches)], dim=1
    ).to(torch.int64)
    patch_positions = (
        torch.stack(
            [
                torch.arange(num_patches) // (224 // patch_size),
                torch.arange(num_patches) % (224 // patch_size),
            ],
            dim=1,
        ).to(torch.int64)
        * patch_size
    )
    from src.models.components.utils.part_utils import compute_gt_transform

    transform = compute_gt_transform(pair_indices[None], patch_positions[None])[0]

    root = 0
    # transforms = get_refpatch_transforms(transform, pair_indices, root)
    # transforms = {k: v + 16 * torch.randn(2) for k, v in transforms.items()}

    # img_abs, img_rel = create_provenance_grid(
    #     patch_positions, root, transforms, patch_size
    # )

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
    # axes[0][0].imshow(img_abs[:, :, 0])
    # axes[0][1].imshow(img_abs[:, :, 1])
    # plot_provenance(axes[1][0], img_abs, abs=True)
    # plot_provenance(axes[1][1], img_rel, abs=False)
    # fig.savefig("example_0.png")
