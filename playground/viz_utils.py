import os
import re
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (AnnotationBbox, OffsetImage)
from matplotlib.patches import Rectangle
import numpy as np
import csv
import torch
import skimage


# artifacts_download_path = os.path.expanduser(artifacts_download_path)

def reconstruct_image_unfold_v2(patches, target, img_size=32, patch_size=4):
    '''
    image is a [3, 32, 32] puzzled image with a certain patch size, their correct position is known by the target.
    target is a [2, #num_patches]
    '''

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    xs, ys = target[0, :], target[1, :]
    # patches = gt_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).reshape(3, -1, patch_size, patch_size).permute(1, 2, 3, 0)  # torch.Size([64, 4, 4, 3])
    recon_img = np.zeros((img_size, img_size, 3))

    num_patches = target.shape[1]  # 64

    for i in range(num_patches):
        recon_img[ys[i]: ys[i] + patch_size, xs[i]: xs[i] + patch_size, :] = patches[i, :, :, :]
        # patches[i, :, :, :] should be placed at xs[i] and ys[i]
        imagebox = OffsetImage(patches[i, :, :, :], zoom=8)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox,
                            (xs[i] + patch_size // 2, img_size - ys[i] - patch_size // 2),  # this position is the middle of the patch position
                            pad=0,
                            # bboxprops=dict(facecolor='none', edgecolor='black')
                            )
        ax.add_artist(ab)

    # Fix the display limits to see everything
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    plt.savefig("./viz_out/myImagePDF1.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    plt.imshow(recon_img.astype(int))
    plt.savefig("./viz_out/myImagePDF2.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def reconstruct_image_unfold_v4(patches, target, img_size=32):
    '''
    image is a [3, 32, 32] puzzled image with a certain patch size, their correct position is known by the target.
    target is a [2, #num_patches]
    '''

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    _, indices = torch.sort((target[2, :] - target[0, :]) * (target[3, :] - target[1, :]), descending=True, stable=True)
    patches = patches[indices, :, :, :]
    target = target[:, indices]

    xs, ys, xe, ye = target[0, :], target[1, :], target[2, :], target[3, :]

    recon_img = np.zeros((img_size, img_size, 3))

    num_patches = target.shape[1]  # 64

    for i in range(num_patches):
        w_i = (xe[i] - xs[i]).item()
        h_i = (ye[i] - ys[i]).item()
        patch = skimage.transform.resize(patches[i, :, :, :], (h_i, w_i, 3), order=0)  # resize to the original patch
        recon_img[ys[i]: ye[i], xs[i]: xe[i], :] = patch
        # patches[i, :, :, :] should be placed at xs[i] and ys[i]
        imagebox = OffsetImage(patch, zoom=8)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox,
                            (xs[i] + w_i / 2, img_size - ys[i] - h_i / 2),  # this position is the middle of the patch position
                            pad=0,
                            # bboxprops=dict(facecolor='none', edgecolor='black')
                            )
        ax.add_artist(ab)

    # Fix the display limits to see everything
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    plt.savefig("./viz_out/myImagePDF1.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    plt.imshow(recon_img.astype(int))
    plt.savefig("./viz_out/myImagePDF2.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def viz_img_bbs(img, target, img_size=32):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # draw the image
    ax.imshow(img)
    xs, ys, xe, ye = target[0, :], target[1, :], target[2, :], target[3, :]
    num_patches = target.shape[1]  # 64

    for i in range(num_patches):
        w_i = (xe[i] - xs[i]).item()
        h_i = (ye[i] - ys[i]).item()
        import matplotlib.colors as mcolors
        ax.add_patch(Rectangle((xs[i] - 0.5, ys[i] - 0.5),  # the image starts from -0.5
                               w_i,
                               h_i,
                               edgecolor=list(mcolors.TABLEAU_COLORS.values())[i % 10],  # there are only 10 colors
                               # edgecolor='orange',
                               facecolor='none',
                               linewidth=2,
                               alpha=0.5
                               ))
    plt.savefig("./viz_out/myImagePDF3.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()