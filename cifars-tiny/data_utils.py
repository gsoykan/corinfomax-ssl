import torch
import torchvision
import transform_utils as utils
import random

from custom_datasets.ssl_comics_crops_dataset import SSLComicsCropsDataset


def make_data(dataset, subset, subset_type):
    """Build data folder, includes defined image augmentations in utils as function.
    """
    if dataset == 'cifar10':
        pretrain_data = torchvision.datasets.CIFAR10(root="data", train=True, \
                                                     transform=utils.PretrainTransform(dataset), download=True)
        lin_train_data = torchvision.datasets.CIFAR10(root="data", train=True, \
                                                      transform=utils.EvalTransform(dataset, train_transform=True),
                                                      download=True)

        lin_test_data = torchvision.datasets.CIFAR10(root="data", train=False, \
                                                     transform=utils.EvalTransform(dataset, train_transform=False),
                                                     download=True)
    if dataset == 'cifar100':
        pretrain_data = torchvision.datasets.CIFAR100(root="data", train=True, \
                                                      transform=utils.PretrainTransform(dataset), download=True)
        lin_train_data = torchvision.datasets.CIFAR100(root="data", train=True, \
                                                       transform=utils.EvalTransform(dataset, train_transform=True),
                                                       download=True)
        lin_test_data = torchvision.datasets.CIFAR100(root="data", train=False, \
                                                      transform=utils.EvalTransform(dataset, train_transform=False),
                                                      download=True)
    if dataset == 'tiny_imagenet':
        pretrain_data = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/train", \
                                                         transform=utils.PretrainTransform(dataset))
        lin_train_data = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/train", \
                                                          transform=utils.EvalTransform(dataset, train_transform=True))
        lin_test_data = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/val", \
                                                         transform=utils.EvalTransform(dataset, train_transform=False))
    if dataset == 'comics_crops_bodies':
        dataset = SSLComicsCropsDataset(
            prefiltered_csv_folder_dir='/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/data/ssl/filtered_all_body.csv',
            transform=utils.PretrainTransform(dataset),
            item_type='body'
        )
        pretrain_data, lin_train_data, lin_test_data = dataset, None, None

    return pretrain_data, lin_train_data, lin_test_data
