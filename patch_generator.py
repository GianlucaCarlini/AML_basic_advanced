# %% import

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# %% patch generation


def patch_generator(
    img_path,
    mask_path,
    savedir_img,
    savedir_masks,
    threshold=0.1,
    pad=True,
    patch_size=256,
    stride=256,
):

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    if ((mask > 0.0).astype("uint8")).sum() > (0.05 * img.shape[0] * img.shape[1]):
        img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(
            mask, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC
        )

    mask = np.where(mask > 0, 255, 0)

    if pad:
        pad_width = int(
            (patch_size * ((img.shape[1] // patch_size) + 1) - img.shape[1]) / 2
        )
        pad_height = int(
            (patch_size * ((img.shape[0] // patch_size) + 1) - img.shape[0]) / 2
        )
        img = np.pad(img, [(pad_width, pad_width), (pad_height, pad_height), (0, 0)])
        mask = np.pad(mask, [(pad_width, pad_width), (pad_height, pad_height)])

    img = img[
        : (img.shape[0] // patch_size) * patch_size,
        : (img.shape[1] // patch_size) * patch_size,
    ]
    mask = mask[
        : (mask.shape[0] // patch_size) * patch_size,
        : (mask.shape[1] // patch_size) * patch_size,
    ]

    N_PATCH_H = 1 + (img.shape[0] - patch_size) // stride
    N_PATCH_W = 1 + (img.shape[1] - patch_size) // stride

    n = 0

    for i in range(N_PATCH_H):
        for j in range(N_PATCH_W):

            savepath_img = os.path.join(savedir_img, os.path.basename(img_path))
            savepath_img = savepath_img[:-4] + f"_{i}_{j}"
            savepath_masks = os.path.join(savedir_masks, os.path.basename(mask_path))
            savepath_masks = savepath_masks[:-4] + f"_{i}_{j}"

            img_patch = img[
                i * stride : i * stride + patch_size,
                j * stride : j * stride + patch_size,
            ]
            mask_patch = mask[
                i * stride : i * stride + patch_size,
                j * stride : j * stride + patch_size,
            ]

            if ((mask_patch > 0.0).astype("uint8")).sum() > (
                threshold * patch_size**2
            ):
                cv2.imwrite(f"{savepath_img}.png", img_patch)
                cv2.imwrite(f"{savepath_masks}.png", mask_patch)
                n += 1

    print(f"generated {n} patches of {os.path.basename(img_path)}")


# %% saving train patches

img_path = "./data/Train/images"
mask_path = "./data/Train/masks"

savepath_img = "./data/Train/images/patches"
savepath_masks = "./data/Train/masks/patches"

imgs = os.listdir(img_path)
masks = os.listdir(mask_path)

for img, mask in zip(imgs, masks):

    if os.path.isdir(os.path.join(img_path, img)):
        continue

    patch_generator(
        os.path.join(img_path, img),
        os.path.join(mask_path, mask),
        savepath_img,
        savepath_masks,
        threshold=0.01,
        pad=False,
        patch_size=256,
        stride=256,
    )

# %%
