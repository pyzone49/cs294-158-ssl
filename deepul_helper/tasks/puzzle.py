# WIP
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepul_helper.batch_norm import BatchNorm1d


class PuzzleSolver(nn.Module):
    latent_dim = 4096
    metrics = ['Loss']
    metrics_fmt = [':.4e']

    def __init__(self, dataset, n_classes):
        super().__init__()
        self.target_dim = 96
        self.emb_scale = 0.1
        self.nb_lines = 7
        self.steps_to_ignore = 1
        self.steps_to_predict = 1
        self.n_classes = n_classes

        self.network = AlexNetwork()
    def construct_classifier(self):
        return nn.Sequential(BatchNorm1d(self.latent_dim, center=False), nn.Linear(self.latent_dim, self.n_classes))

    def forward(self,  uniform_patch, random_patch, labels):
        output_fc6_uniform = self.network.forward_once(uniform_patch)
        output_fc6_random = self.network.forward_once(random_patch)
        output = torch.cat((output_fc6_uniform,output_fc6_random), 1)
        output = self.network.fc(output)
        return output, output_fc6_uniform, output_fc6_random


def images_to_cpc_patches(images, patch_size=96, stride=4, grid_dim=(3, 3)):
    all_image_patches = []
    all_labels = []
    for r in range(grid_dim[0]):
        for c in range(grid_dim[1]):

            # Extract a patch of size (patch_size, patch_size) at the (r, c) location
            start_y = r * stride
            start_x = c * stride
            batch_patch = images[:, :, start_y:start_y + patch_size, start_x:start_x + patch_size]
            all_image_patches.append(batch_patch)

            # Create labels representing the (row, column) position in the grid
            labels = torch.tensor([r, c], dtype=torch.long).repeat(images.size(0), 1)  # Repeat for the batch
            all_labels.append(labels)

    # Stack patches and labels along a new dimension
    image_patches_tensor = torch.stack(all_image_patches, dim=1)
    labels_tensor = torch.cat(all_labels, dim=0)

    # Reshape patches and labels for CPC training
    patches = image_patches_tensor.view(-1, *image_patches_tensor.shape[-3:])
    return patches, labels_tensor


class AlexNetwork(nn.Module):
    def __init__(self, aux_logits=False):
        super(AlexNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(96),

            nn.Conv2d(96, 384, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(384),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * 4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 8)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc6(output)
        return output

    def forward(self, uniform_patch, random_patch):
        output_fc6_uniform = self.forward_once(uniform_patch)
        output_fc6_random = self.forward_once(random_patch)
        output = torch.cat((output_fc6_uniform, output_fc6_random), 1)
        output = self.fc(output)
        return output, output_fc6_uniform, output_fc6_random
