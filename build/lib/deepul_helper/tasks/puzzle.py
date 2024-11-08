import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage


class PuzzleSolver(nn.Module):
    metrics = ['Loss']
    metrics_fmt = [':.4e']

    def __init__(self, dataset, n_classes):
        super().__init__()
        input_channels = 3

        self.latent_dim = 4000
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
        self.dataset = dataset
        self.n_classes = n_classes

    def get_patch_from_grid(self, image, patch_dim, gap):
        image = np.array(image)

        offset_x, offset_y = image.shape[0] - (patch_dim * 3 + gap * 2), image.shape[1] - (patch_dim * 3 + gap * 2)
        start_grid_x, start_grid_y = np.random.randint(0, offset_x), np.random.randint(0, offset_y)
        patch_loc_arr = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
        loc = np.random.randint(len(patch_loc_arr))
        tempx, tempy = patch_loc_arr[loc]

        patch_x_pt = start_grid_x + patch_dim * (tempx - 1) + gap * (tempx - 1)
        patch_y_pt = start_grid_y + patch_dim * (tempy - 1) + gap * (tempy - 1)
        random_patch = image[patch_x_pt:patch_x_pt + patch_dim, patch_y_pt:patch_y_pt + patch_dim]

        patch_x_pt = start_grid_x + patch_dim * (2 - 1) + gap * (2 - 1)
        patch_y_pt = start_grid_y + patch_dim * (2 - 1) + gap * (2 - 1)
        uniform_patch = image[patch_x_pt:patch_x_pt + patch_dim, patch_y_pt:patch_y_pt + patch_dim]

        random_patch_label = loc
        return uniform_patch, random_patch, random_patch_label

    def forward(self, image):
        # Perform preprocessing to get patches
        uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(image, self.patch_dim, self.gap)

        # Resize patches if necessary
        if uniform_patch.shape[0] != 96:
            uniform_patch = skimage.transform.resize(uniform_patch, (96, 96))
            random_patch = skimage.transform.resize(random_patch, (96, 96))

            uniform_patch = skimage.img_as_float32(uniform_patch)
            random_patch = skimage.img_as_float32(random_patch)

        # Add noise to channels 2 and 3
        for patch in [uniform_patch, random_patch]:
            patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(patch[:, :, 0]), patch[:, :, 1].shape)
            patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(patch[:, :, 0]), patch[:, :, 2].shape)

        # Convert label to tensor
        random_patch_label = torch.tensor(random_patch_label).long()
        # Apply transform if available
        if self.transform:
            uniform_patch = self.transform(uniform_patch)
            random_patch = self.transform(random_patch)

        output_fc6_uniform = self.forward_once(uniform_patch)
        output_fc6_random = self.forward_once(random_patch)
        output = torch.cat((output_fc6_uniform, output_fc6_random), 1)
        output = self.fc(output)
        return output, output_fc6_uniform, output_fc6_random

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc6(output)
        return output

    def construct_classifier(self):
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.latent_dim, affine=False),
            nn.Linear(self.latent_dim, self.n_classes)
        )
        return classifier