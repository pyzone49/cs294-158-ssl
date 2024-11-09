import os.path as osp
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
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
        self.patch_size = 96
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
        #save img

        #resize
        image = skimage.transform.resize(image, (400, 400))
        # Convert pixel values back to [0, 255] range and to uint8
        image = (image * 255).astype(np.uint8)
        # cv2.imwrite(
        #     "/Users/yacineflici/Documents/master-vmi/s3/IFLCM010 Analyse d'images/TP5/self-supervised-learning/cs294-158-ssl/img.jpg",
        #     image)
        all_image_patches = []
        all_labels = []
        for r in range(patch_dim[0]):
            for c in range(patch_dim[1]):
                # Extract a patch of size (patch_size, patch_size) at the (r, c) location
                start_y = r * (self.patch_size + gap)
                start_x = c * (self.patch_size + gap)
                batch_patch = image[start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]
                #save patch
                # cv2.imwrite(f"/Users/yacineflici/Documents/master-vmi/s3/IFLCM010 Analyse d'images/TP5/self-supervised-learning/cs294-158-ssl/patch_{r}_{c}.jpg", batch_patch)
                all_image_patches.append(batch_patch)
                all_labels.append(r*patch_dim[1] + c)
        #remove center patch label
        all_labels.pop(patch_dim[0] * patch_dim[1] // 2)
        #choose a random patch
        random_patch_label = random.choice(all_labels)
        random_patch = all_image_patches[random_patch_label]
        #choose a uniform patch
        uniform_patch = all_image_patches[patch_dim[0] * patch_dim[1] // 2]
        #all to PIL
        uniform_patch = Image.fromarray((uniform_patch * 255).astype(np.uint8))
        random_patch = Image.fromarray((random_patch * 255).astype(np.uint8))
        #save uniform patch
        # uniform_patch.save("/Users/yacineflici/Documents/master-vmi/s3/IFLCM010 Analyse d'images/TP5/self-supervised-learning/cs294-158-ssl/uniform_patch.jpg")
        #save random patch
        # random_patch.save("/Users/yacineflici/Documents/master-vmi/s3/IFLCM010 Analyse d'images/TP5/self-supervised-learning/cs294-158-ssl/random_patch.jpg")
        #labels to tensor
        random_patch_label = torch.tensor(random_patch_label, dtype=torch.long)
        return uniform_patch, random_patch, random_patch_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # image = Image.open(self.data[index][0]).convert('RGB')
        image = self.data[index][0]
        uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(image,
                                                                                   self.patch_dim,
                                                                                   self.gap)
        if self.transform:
            uniform_patch = self.transform(uniform_patch)
            random_patch = self.transform(random_patch)
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
        transform = transforms.Compose([
                                        # transforms.Resize((400, 400)),
                                        #to gray
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
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
        train_dset = PuzzleDataset(dataset, patch_dim=(3, 3), gap=10, validate=False, transform=get_transform(dataset, task, train=True))
        test_dset = PuzzleDataset(dataset, patch_dim=(3, 3), gap=10, validate=True, transform=get_transform(dataset, task, train=False))
        # train_dset = datasets.CIFAR10(osp.join('data', dataset), train=True, download=True,transform=get_transform(dataset, task, train=True))
        # test_dset = datasets.CIFAR10(osp.join('data', dataset), train=False, download=True,transform=get_transform(dataset, task, train=False))
        # train_dset = datasets.VOCDetection(root=osp.join('data', dataset), year='2012', image_set='train', download=True,
        #                                    transform=get_transform(dataset, task, train=True))
        # test_dset = datasets.VOCDetection(osp.join('data', dataset), year='2012', image_set='val', download=True, transform=get_transform(dataset, task, train=False))

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
