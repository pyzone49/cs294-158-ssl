import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from deepul_helper.demos import load_model_and_data
from deepul_helper.data import get_transform

if __name__ == '__main__':
    # Load model and data
    model, linear_classifier, train_loader, test_loader = load_model_and_data('puzzle')

    # Get a single batch from the loader
    data_iter = iter(train_loader)
    uniform_patch, random_patch, random_patch_label = next(data_iter)

    # Convert patches to numpy arrays for visualization
    uniform_patch_np = uniform_patch[0].permute(1, 2, 0).numpy()  # Convert [1, C, H, W] -> [H, W, C]
    random_patch_np = random_patch[0].permute(1, 2, 0).numpy()  # Convert [1, C, H, W] -> [H, W, C]

    # Display the patches
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(uniform_patch_np)
    plt.title('Uniform Patch')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(random_patch_np)
    # plt.title(f'Random Patch (Label:  {random_patch_label.item()})')
    plt.axis('off')
    plt.show()