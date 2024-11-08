import os.path as osp
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision import datasets, transforms
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import skimage
from skimage import transform, img_as_float32

class PuzzleDataset(Dataset):
    def __init__(self, dataset, patch_dim, gap, validate=False, transform=None):
        """
        Initialize PuzzleDataset with a dataset string specifying the type of dataset.

        :param dataset: The dataset type as a string ('cifar10', 'pascalvoc2012', 'fashionmnist').
        :param patch_dim: The dimension of each patch.
        :param gap: The gap between patches in the grid.
        :param validate: Boolean to switch between train and validation sets.
        :param transform: Optional transform to be applied on a patch.
        """
        self.patch_dim = patch_dim
        self.gap = gap
        self.transform = transform

        # Load the specified dataset based on the string value of `dataset`
        if dataset == 'cifar10':
            self.data = datasets.CIFAR10(
                osp.join('data', dataset),
                train=not validate,
                transform=None,  # We'll apply transformations in __getitem__
                download=True
            )
        elif dataset == 'pascalvoc2012':
            self.data = datasets.VOCSegmentation(
                osp.join('data', dataset),
                image_set='val' if validate else 'train',
                transforms=None,  # No built-in transforms; we'll handle them manually
                download=True
            )
        elif dataset == 'fashionmnist':
            self.data = datasets.FashionMNIST(
                osp.join('data', dataset),
                train=not validate,
                transform=None,  # We'll apply transformations in __getitem__
                download=True
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # image = Image.open(self.data[index][0]).convert('RGB')
        image = self.data[index][0]
        uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(image,
                                                                                   self.patch_dim,
                                                                                   self.gap)
        if uniform_patch.shape[0] != 96:
            uniform_patch = skimage.transform.resize(uniform_patch, (96, 96))
            random_patch = skimage.transform.resize(random_patch, (96, 96))

            uniform_patch = img_as_float32(uniform_patch)
            random_patch = img_as_float32(random_patch)

        # Dropped color channels 2 and 3 and replaced with gaussian noise(std ~1/100 of the std of the remaining channel)
        uniform_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]),
                                                  (uniform_patch.shape[0], uniform_patch.shape[1]))
        uniform_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]),
                                                  (uniform_patch.shape[0], uniform_patch.shape[1]))
        random_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]),
                                                 (random_patch.shape[0], random_patch.shape[1]))
        random_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]),
                                                 (random_patch.shape[0], random_patch.shape[1]))

        random_patch_label = np.array(random_patch_label).astype(np.int64)

        if self.transform:
            uniform_patch = self.transform(uniform_patch)
            random_patch = self.transform(random_patch)

        #save image if index < 10
        if index < 10:
            # Convert tensors to numpy arrays
            uniform_patch_np = uniform_patch.permute(1, 2, 0).cpu().numpy()  # Convert [C, H, W] -> [H, W, C]
            random_patch_np = random_patch.permute(1, 2, 0).cpu().numpy()  # Convert [C, H, W] -> [H, W, C]

            plt.imsave(f"uniform_patch_{index}.png", uniform_patch_np.permute(1, 2, 0))
            plt.imsave(f"random_patch_{index}.png", random_patch_np.permute(1, 2, 0))
        return uniform_patch, random_patch, random_patch_label


def get_transform(dataset, task, train=True):
    print(f"get_transform({dataset}, {task}, {train})")
    transform = None
    if task == 'context_encoder':
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        elif 'imagenet' in dataset:
            transform = transforms.Compose([
                transforms.Resize(350),
                transforms.RandomCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif dataset == 'fashionmnist':
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

    elif task == "puzzle":
        transform =  transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225])])
    elif task == 'rotation':
        if dataset == 'cifar10':
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        elif 'imagenet' in dataset:
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        elif dataset == 'fashionmnist':
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
            else:
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
    elif task == 'cpc':
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    elif task == 'simclr':
        if dataset == 'cifar10':
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
        elif 'imagenet' in dataset:
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=11),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(128),
                    transforms.CenterCrop(128),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        elif dataset == 'fashionmnist':
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ])
        transform = SimCLRDataTransform(transform)
    elif task == 'segmentation':
        if train:
            transform = MultipleCompose([
                MultipleRandomResizedCrop(128),
                MultipleRandomHorizontalFlip(),
                RepeatTransform(transforms.ToTensor()),
                GroupTransform([
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    SegTargetTransform()])
            ])
        else:
            transform = MultipleCompose([
                RepeatTransform(transforms.Resize(128)),
                RepeatTransform(transforms.CenterCrop(128)),
                RepeatTransform(transforms.ToTensor()),
                GroupTransform([
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    SegTargetTransform()])
            ])

    else:
        raise Exception('Invalid task:', task)

    return transform


def get_datasets(dataset, task):
    if task == "puzzle":
        train_dset = PuzzleDataset(dataset,patch_dim=5, gap=1, validate=False,
                                   transform=get_transform(dataset, task, train=True))
        test_dset = PuzzleDataset(dataset,patch_dim=5, gap=1, validate=True,
                                  transform=get_transform(dataset, task, train=False))
        return train_dset, test_dset, 10

    if 'imagenet' in dataset:
        train_dir = osp.join('data', dataset, 'train')
        val_dir = osp.join('data', dataset, 'val')
        train_dataset = datasets.ImageFolder(
            train_dir,
            get_transform(dataset, task, train=True)
        )

        val_dataset = datasets.ImageFolder(
            val_dir,
            get_transform(dataset, task, train=False)
        )

        return train_dataset, val_dataset, len(train_dataset.classes)
    elif dataset == 'cifar10':
        train_dset = datasets.CIFAR10(osp.join('data', dataset), train=True,
                                      transform=get_transform(dataset, task, train=True),
                                      download=True)
        test_dset = datasets.CIFAR10(osp.join('data', dataset), train=False,
                                     transform=get_transform(dataset, task, train=False),
                                     download=True)
        return train_dset, test_dset, len(train_dset.classes)
    elif dataset == 'pascalvoc2012':
        train_dset = datasets.VOCSegmentation(osp.join('data', dataset), image_set='train',
                                              transforms=get_transform(dataset, task, train=True),
                                              download=True)
        test_dset = datasets.VOCSegmentation(osp.join('data', dataset), image_set='val',
                                             transforms=get_transform(dataset, task, train=False),
                                             download=True)
        return train_dset, test_dset, 21
    elif dataset == "fashionmnist":
        train_dset = datasets.FashionMNIST(osp.join('data', dataset), train=True,
                                           transform=get_transform(dataset, task, train=True),
                                           download=True)
        test_dset = datasets.FashionMNIST(osp.join('data', dataset), train=False,
                                          transform=get_transform(dataset, task, train=False),
                                          download=True)
        return train_dset, test_dset, 10
    else:
        raise Exception('Invalid dataset:', dataset)


# https://github.com/sthalles/SimCLR/blob/master/data_aug/gaussian_blur.py
class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

# Re-written torchvision transforms to support operations on multiple inputs
# Needed to maintain consistency on random transforms with real images and their segmentations
class MultipleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        for t in self.transforms:
            inputs = t(*inputs)
        return inputs


class GroupTransform(object):
    """ Applies a list of transforms elementwise """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        assert len(inputs) == len(self.transforms)
        outputs = [t(inp) for t, inp in zip(self.transforms, inputs)]
        return outputs

class MultipleRandomResizedCrop(transforms.RandomResizedCrop):

    def __call__(self, *imgs):
        """
        Args:
            imgs (List of PIL Image): Images to be cropped and resized.
                                      Assumes they are all the same size

        Returns:
            PIL Images: Randomly cropped and resized images.
        """
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
                for img in imgs]

class MultipleRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, *imgs):
        if random.random() < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs

class RepeatTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, *inputs):
        return [self.transform(inp) for inp in inputs]

class SegTargetTransform(object):
    def __call__(self, target):
        target *= 255.
        target[target > 20] = 0
        return target.long()
