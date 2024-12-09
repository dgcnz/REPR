import torch
import time
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from jaxtyping import Float, Int
from torch import Tensor
import numpy as np
import skimage.transform
import math
from torchvision import transforms
from timm.data import create_transform
from collections import namedtuple
from torchvision.datasets import CIFAR10
import timeit
from functools import partial
import torch.nn.functional as F



def patchify_and_resize(start_x, start_y, end_x, end_y, image, resize):
    # print(image[:, start_y:end_y, start_x:end_x].shape)
    image_patched = image[:, start_y:end_y, start_x:end_x]
    if resize is not None:
        image_patched = skimage.transform.resize(image_patched, resize)

    return image_patched



class MelikaSampler(object):
    def __init__(self, img: Image.Image, patch_size: int):
        assert (
            img.width == img.height
        ), f"Image should be square, got {img.width}x{img.height}"
        img_size = img.height
        x_start = torch.arange(img_size - patch_size + 1)
        y_start = torch.arange(img_size - patch_size + 1)
        # save x_start and y_start in a file for future loading
        t = torch.cartesian_prod(
            y_start, x_start
        ).T  # [y_start, x_start] torch.Size([2, 841])
        targets = torch.cat(
            (t[None, 1, :], t[None, 0, :]), dim=0
        )  # [2, 841] switch y and x to x and y

        self.img = img
        self.patch_size = patch_size
        self.targets = targets

    def __call__(self):
        img = self.img
        img_size = img.height
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        rand_ind = torch.randint(
            0, self.targets.shape[1], (num_patches,)
        )  # torch.Size([64])
        target = self.targets[:, rand_ind]  # [num_channels, 64]: chosen x1, y1, x2, y2

        # if transform is not None:
        #     img = transform(img)  # [3, 32, 32]
        img = to_tensor(img)
        patchify = np.vectorize(
            patchify_and_resize,
            excluded=["image", "resize"],
            # otypes=['uint8'],
            signature="(),(),(),()->(l,m,k)",
        )

        img_patches = patchify(
            target[0],
            target[1],
            target[0] + patch_size,
            target[1] + patch_size,
            image=np.asarray(img),
            resize=None,
        )
        patches_per_width = int(math.sqrt(num_patches))
        img_patches = torch.from_numpy(img_patches)
        # [3, 64, 4, 4] -> [3, 8, 8, 4, 4] -> [3, 8, 4, 8, 4]
        img_patches = img_patches.permute(1, 0, 2, 3)
        img_patches = img_patches.reshape(
            3, patches_per_width, patches_per_width, patch_size, patch_size
        )
        img_patches = img_patches.permute(0, 1, 3, 2, 4).reshape(3, img_size, img_size)
        return img_patches


def grid_sample_loop(
    img: Float[Tensor, "B C H W"], patch_size: int
) -> Float[Tensor, "B C H W"]:
    B, C, H, W = img.size()
    patched_dims = (H // patch_size, W // patch_size)
    n_patches = patched_dims[0] * patched_dims[1]
    ys = torch.randint(0, H - patch_size, (B, n_patches))
    xs = torch.randint(0, W - patch_size, (B, n_patches))
    output = torch.zeros((B, C, H, W))

    for b in range(B):
        for r in range(patched_dims[0]):
            for c in range(patched_dims[1]):
                patch_idx = r * patched_dims[1] + c
                x_src = xs[b, patch_idx]
                y_src = ys[b, patch_idx]
                x_dst = c * patch_size
                y_dst = r * patch_size
                output[b, :, y_dst : y_dst + patch_size, x_dst : x_dst + patch_size] = (
                    img[b, :, y_src : y_src + patch_size, x_src : x_src + patch_size]
                )

    return output



def create_grid_batched(
    patch_size: int, H: int, W: int, ys: Int[Tensor, "B N"], xs: Int[Tensor, "B N"]
):
    B, nH, nW = ys.size(0), H // patch_size, W // patch_size
    dx = torch.arange(patch_size, device=ys.device)
    dy = torch.arange(patch_size, device=ys.device)
    gpy, gpx = torch.meshgrid(dy, dx, indexing="ij")
    yy = ys[:, :, None, None] + gpy.unsqueeze(0)
    xx = xs[:, :, None, None] + gpx.unsqueeze(0)

    yy = yy.view(B, nH, nW, patch_size, patch_size)
    yy = yy.permute(0, 1, 3, 2, 4).reshape(B, H, W)
    xx = xx.view(B, nH, nW, patch_size, patch_size)
    xx = xx.permute(0, 1, 3, 2, 4).reshape(B, H, W)
    return yy, xx

def create_grid(
    patch_size: int, H: int, W: int, ys: Int[Tensor, "N"], xs: Int[Tensor, "N"]
):
    nH, nW = H // patch_size, W // patch_size
    dx = torch.arange(patch_size, device=ys.device)
    dy = torch.arange(patch_size, device=ys.device)
    gpy, gpx = torch.meshgrid(dy, dx, indexing="ij")
    yy = ys[:, None, None] + gpy.unsqueeze(0)
    xx = xs[:, None, None] + gpx.unsqueeze(0)

    yy = yy.view(nH, nW, patch_size, patch_size)
    yy = yy.permute(0, 2, 1, 3).reshape(H, W)
    xx = xx.view(nH, nW, patch_size, patch_size)
    xx = xx.permute(0, 2, 1, 3).reshape(H, W)
    return yy, xx

def grid_sample_custom(
    img: Float[Tensor, "B C H W"], patch_size: int, sampler="ongrid", batch_impl="arange"
):
    B, C, H, W = img.size()
    if sampler == "ongrid":
        ys, xs = sample_ongrid(B, H, W, patch_size, device=img.device)
    if sampler == "canonical":
        ys, xs = sample_ongrid(B, H, W, patch_size, canonical=True, device=img.device)
    else:
        ys, xs = sample_offgrid(B, H, W, patch_size, device=img.device)

    if batch_impl == "arange":
        grid_y, grid_x = create_grid_batched(patch_size, H, W, ys, xs)
        grid_y = grid_y[:, None, :, :].expand(-1, C, -1, -1)  # Shape: (B, C, H', W')
        grid_x = grid_x[:, None, :, :].expand(-1, C, -1, -1)
        # Create batch and channel indices
        batch_indices = torch.arange(B, device=img.device)[:, None, None, None].expand(-1, C, grid_y.shape[2], grid_y.shape[3])
        channel_indices = torch.arange(C, device=img.device)[None, :, None, None].expand(B, -1, grid_y.shape[2], grid_y.shape[3])
        # Use advanced indexing to sample the image
        output = img[batch_indices, channel_indices, grid_y, grid_x]
        assert output.shape == (B, C, H, W)
        return output
    elif batch_impl == "gather":
        grid_y, grid_x = create_grid_batched(patch_size, H, W, ys, xs)  # Shapes: (B, H', W')

        # Flatten the grids
        grid_y_flat = grid_y.reshape(B, -1)  # Shape: (B, H'*W')
        grid_x_flat = grid_x.reshape(B, -1)  # Shape: (B, H'*W')

        # Compute linear indices
        indices = grid_y_flat * W + grid_x_flat  # Shape: (B, H'*W')

        # Flatten the image
        img_flat = img.view(B, C, H * W)  # Shape: (B, C, H*W)

        # Gather pixel values using the computed indices
        output = torch.gather(img_flat, 2, indices.unsqueeze(1).expand(-1, C, -1))  # Shape: (B, C, H'*W')

        # Reshape to the original spatial dimensions
        output = output.view(B, C, H, W)

        return output
    elif batch_impl == "gather_unflattened":
        grid_y, grid_x = create_grid_batched(
            patch_size, H, W, ys, xs
        )  # Shapes: (B, H', W')

        # Compute linear indices
        indices = grid_y * W + grid_x  # Shape: (B, H', W')
        indices = indices.view(B, -1)  # Shape: (B, H'*W')

        # Flatten the image
        img_flat = img.view(B, C, H * W)  # Shape: (B, C, H*W)

        # Gather pixel values
        output = img_flat.gather(2, indices.unsqueeze(1).expand(-1, C, -1))  # Shape: (B, C, H'*W')

        # Reshape to original spatial dimensions
        H_prime, W_prime = grid_y.size(1), grid_x.size(2)
        output = output.view(B, C, H_prime, W_prime)

        return output
    elif batch_impl == "loop_batched":
        output = torch.zeros((B, C, H, W), device=img.device)
        grid_y, grid_x = create_grid_batched(patch_size, H, W, ys, xs)
        for b in range(B):
            output[b, :, :, :] = img[b, :, grid_y[b], grid_x[b]]
        return output
    elif batch_impl == "loop":
        output = torch.zeros((B, C, H, W), device=img.device)
        for b in range(B):
            grid_y, grid_x = create_grid(patch_size, H, W, ys[b], xs[b])
            output[b, :, :, :] = img[b, :, grid_y, grid_x]
        return output
    elif batch_impl == "torch.grid_sample":
        grid_y, grid_x = create_grid_batched(patch_size, H, W, ys, xs)
        # Normalize grid coordinates to [-1, 1]
        grid_y = (grid_y.float() / (H - 1)) * 2 - 1
        grid_x = (grid_x.float() / (W - 1)) * 2 - 1

        # Stack grid_x and grid_y to create the sampling grid
        grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: [B, H, W, 2]

        # Sample the image using the grid
        output = F.grid_sample(img, grid, align_corners=True)
        return output


def sample_offgrid(
    B: int, H: int, W: int, patch_size: int, device: torch.device = "cpu"
) -> tuple[Int[Tensor, "B N"], Int[Tensor, "B N"]]:
    nW = W // patch_size
    nH = H // patch_size
    xs = torch.randint(0, H - patch_size, (B, nH * nW), device=device)
    ys = torch.randint(0, W - patch_size, (B, nH * nW), device=device)
    return ys, xs


def sample_ongrid(
    B: int,
    H: int,
    W: int,
    patch_size: int,
    canonical: bool = False,
    device: torch.device = "cpu",
) -> tuple[Int[Tensor, "B N"], Int[Tensor, "B N"]]:
    nW = W // patch_size
    nH = H // patch_size
    idx = torch.arange(nH * nW, device=device)
    if not canonical:
        idx = idx[torch.randperm(nH * nW, device=device)]
    idx = idx.repeat(B, 1)
    xs = idx % nW
    ys = idx // nW
    xs = xs * patch_size
    ys = ys * patch_size
    return ys, xs

def benchmark_cuda(f, *args, **kwargs):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    f(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def benchmark_cpu(f, *args, **kwargs):
    start = time.time()
    f(*args, **kwargs)
    return time.time() - start




patch_size = 4

dataset = CIFAR10(root="data/", download=True, train=False)
img, _ = dataset[0]

sampler = MelikaSampler(img, patch_size)

img.save("grid_img.png")

# time sampler
out = sampler()
print(out.shape)
transforms.ToPILImage()(out).save("grid_melika.png")
t = timeit.timeit(sampler, number=100)
print(t)


img_tensor = to_tensor(img).unsqueeze(0)
# concat to batch 2
img_tensor = torch.cat([img_tensor, img_tensor], dim=0)

# sampler3 = partial(grid_sample_custom, img_tensor, patch_size, "offgrid")
mode = "ongrid"
torch.manual_seed(0)
out_loop = grid_sample_custom(img_tensor.cpu(), patch_size, mode, "loop")
batch_impl = ["gather", "gather_unflattened", "arange", "loop", "torch.grid_sample",  "loop_batched"]
for batch_impl in batch_impl:
    print(batch_impl)
    torch.manual_seed(0)
    sampler3 = partial(grid_sample_custom, img_tensor.cpu(), patch_size, mode, batch_impl)
    out = sampler3()
    assert torch.allclose(out, out_loop)
    transforms.ToPILImage()(out[0]).save(f"grid_meshgrid_{batch_impl}.png")
    t = timeit.timeit(sampler3, number=100)
    print(t)


# for batch_impl in batch_impl:
#     print(batch_impl)
#     torch.manual_seed(0)
#     sampler3 = partial(grid_sample_custom, img_tensor.cpu(), patch_size, mode, batch_impl)
#     sampler3 = torch.compile(grid_sample_custom)
#     out = sampler3(img_tensor
#     assert torch.allclose(out, out_loop)
#     transforms.ToPILImage()(out[0]).save(f"grid_meshgrid_{batch_impl}.png")
#     t = timeit.timeit(sampler3, number=100)
#     print(t)


# sampler2 = partial(grid_sample_loop, img_tensor, patch_size)
# out = sampler2()
# print(out.shape)
# transforms.ToPILImage()(out[0]).save("grid_loop.png")
# t = timeit.timeit(sampler2, number=100)
# print(t)
