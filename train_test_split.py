# %% Import

import os
import shutil
import numpy as np

# %% Get Images

test_size = 0.10

classes_dir = ["images", "masks"]

for cl in classes_dir:

    os.makedirs(f"./data/Train/{cl}")
    os.makedirs(f"./data/Test/{cl}")

all_images = os.listdir("./data/images")

np.random.seed(42)
np.random.shuffle(all_images)

train_images, test_images = np.split(
    all_images, [int(len(all_images) * (1 - test_size))]
)

# %% Copy Images

for cl in classes_dir:
    if cl == "images":
        for img in train_images:
            shutil.copy(f"./data/{cl}/{img}", f"./data/Train/{cl}/{img}")
    else:
        for img in train_images:
            shutil.copy(
                f"./data/{cl}/{img[:-4]}.png", f"./data/Train/{cl}/{img[:-4]}.png"
            )

for cl in classes_dir:
    if cl == "images":
        for img in test_images:
            shutil.copy(f"./data/{cl}/{img}", f"./data/Test/{cl}/{img}")
    else:
        for img in test_images:
            shutil.copy(
                f"./data/{cl}/{img[:-4]}.png", f"./data/Test/{cl}/{img[:-4]}.png"
            )
# %%
