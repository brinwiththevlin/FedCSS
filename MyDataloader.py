import copy
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, WeightedRandomSampler
import random


def uniform_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.ones((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                corruption_matrix[i, j] = 1 - corruption_ratio
            else:
                corruption_matrix[i, j] = corruption_ratio / (num_classes - 1)
    return corruption_matrix


def flip1_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix


def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix


def build_dataset(dataset_name):
    data_train = None
    data_test = None
    num_classes = 0
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        data_train = MNIST(root='data', train=True, transform=transform, download=True)
        data_test = MNIST(root='data', train=False, transform=transform, download=True)
        num_classes = 10
    elif dataset_name == 'cifar10':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        data_train = torchvision.datasets.CIFAR10(root='data', train=True, download=True,
                                                  transform=train_transforms)
        data_test = torchvision.datasets.CIFAR10(root='data', train=False, transform=test_transforms)
        num_classes = 10
    elif dataset_name == 'cifar100':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        data_train = torchvision.datasets.CIFAR100(root='data', train=True, download=True,
                                                  transform=train_transforms)
        data_test = torchvision.datasets.CIFAR100(root='data', train=False, transform=test_transforms)
        num_classes = 100

    return data_train, data_test, num_classes


def load_client_data(dataset_name, client_num, batch_size):
    # Build data
    data_train, data_test, num_classes = build_dataset(dataset_name)

    # Split data into dict
    data_dict = dict()
    train_per_client = len(data_train) // client_num
    test_per_client = len(data_test) // client_num

    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([
                    data_train[i]
                    for i in range((client_idx - 1) *
                                   train_per_client, client_idx * train_per_client)
                ],
                    batch_size,
                    shuffle=True),
            'val':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    batch_size,
                    shuffle=False),
            'test':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    batch_size,
                    shuffle=False)
        }
        data_dict[client_idx - 1] = dataloader_dict

    return data_dict


def load_server_data(args):
    # Build data
    data_train, data_test, num_classes = build_dataset(args.dataset_name)

    num_meta_total = args.validation_num
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    index_to_train = []

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[:]

        index_to_train.extend(index_to_class_for_train)

    random.shuffle(index_to_meta)

    meta_dataset = copy.deepcopy(data_train)
    data_train.data = data_train.data[index_to_train]

    data_train.targets = list(np.array(data_train.targets)[index_to_train])

    meta_dataset.data = meta_dataset.data[index_to_meta]
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    server_dataloader_dict = {
        'train':
            DataLoader(meta_dataset, min(args.batch_size, num_meta_total), shuffle=False,
                       collate_fn=None),
        'val':
            DataLoader(data_test, args.batch_size, shuffle=False,
                       collate_fn=None)
    }

    return server_dataloader_dict


def load_client_weight_data(dataset_name, client_num, batch_size, weight, client_index, loader):
    dataset = copy.deepcopy(loader.dataset)
    dataloader = DataLoader(dataset, batch_size, sampler=WeightedRandomSampler(weight, len(weight)),
                            collate_fn=None)

    return dataloader


def load_corrupt_client_data(
        args,
        client_num,
        imbalanced_factor=None,
        corruption_type=None,
        corruption_ratio=0.,
        corrupt_num=0):
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    # Build data
    data_train, data_test, num_classes = build_dataset(args.dataset_name)

    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num
    num_meta_total = test_per_client

    index_to_train = []

    if imbalanced_factor is not None:
        imbalanced_num_list = []
        sample_num = int((len(data_train.targets) - num_meta_total) / num_classes)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / (imbalanced_factor ** (class_index / (num_classes - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)
        print(imbalanced_num_list)
    else:
        imbalanced_num_list = None

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_class_for_train = index_to_class[:]

        if imbalanced_num_list is not None:
            index_to_class_for_train = index_to_class_for_train[
                                       :min(imbalanced_num_list[class_index], len(index_to_class_for_train))]

        index_to_train.extend(index_to_class_for_train)
    train_per_client = len(index_to_train) // client_num

    np.random.shuffle(index_to_train)
    data_train.data = data_train.data[index_to_train]
    data_train.targets = list(np.array(data_train.targets)[index_to_train])

    targets_true = copy.deepcopy(data_train.targets)

    if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
        print(corruption_matrix)
        if corrupt_num == -1:
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)
        else:
            for index in range(0, corrupt_num * train_per_client):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)


    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([
                    data_train[i]
                    for i in range((client_idx - 1) *
                                   train_per_client, client_idx * train_per_client)
                ], batch_size=args.batch_size, shuffle=False,
                    collate_fn=None),
            'train_targets_true': [
                targets_true[i]
                for i in range((client_idx - 1) *
                               train_per_client, client_idx * train_per_client)
            ],
            'meta_train': [],
            'val':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    args.batch_size,
                    shuffle=False,
                    collate_fn=None),
            'test':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    args.batch_size,
                    shuffle=False,
                    collate_fn=None)
        }
        data_dict[client_idx - 1] = dataloader_dict

    return data_dict


# every client have noniid_ratio of one class, remain of this class give averagely to other clients
def load_non_iid_data(args,
                      client_num,
                      corruption_type=None,
                      corruption_ratio=0.,
                      corrupt_num=0):
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    # Build data
    data_train, data_test, num_classes = build_dataset(args.dataset_name)

    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num

    client_train_index = [[] for i in range(client_num)]
    main_ratio = args.noniid_ratio

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        total_num = len(index_to_class)
        main_num = int(total_num * main_ratio)
        other_num = round(float(total_num - main_num) / (client_num - 1))

        client_train_index[class_index % client_num].extend(index_to_class[0:main_num])
        cnt = 0
        prev_idx = main_num
        for client_idx in range(client_num):
            if client_idx != class_index:
                cnt += 1
                if cnt != client_num - 1:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:prev_idx + other_num])
                    prev_idx += other_num
                else:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:])

    for client_idx in range(client_num):
        np.random.shuffle(client_train_index[client_idx])

    targets_true = copy.deepcopy(data_train.targets)
    if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
        print(corruption_matrix)
        if corrupt_num == -1:
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)
        else:
            for client_idx in range(corrupt_num):
                for index in client_train_index[client_idx]:
                    p = corruption_matrix[int(data_train.targets[index])]
                    data_train.targets[index] = np.random.choice(num_classes, p=p)

    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([
                    data_train[i] for i in client_train_index[client_idx - 1]
                ], batch_size=args.batch_size, shuffle=False,
                collate_fn=None),
            'train_targets_true': [
                targets_true[i] for i in client_train_index[client_idx - 1]],
            'meta_train': [],
            'val':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    args.batch_size,
                    shuffle=False,
                collate_fn=None),
            'test':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    args.batch_size,
                    shuffle=False,
                collate_fn=None)
        }
        data_dict[client_idx - 1] = dataloader_dict

    return data_dict


# every client have noniid_class_num classes
def load_non_iid_class_data(args,
                      client_num,
                      corruption_type=None,
                      corruption_ratio=0.,
                      corrupt_num=0):
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    # Build data
    data_train, data_test, num_classes = build_dataset(args.dataset_name)

    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num

    client_train_index = [[] for i in range(client_num)]
    noniid_class_num = int(num_classes * args.noniid_class_ratio)
    client_per_class = int(client_num * noniid_class_num / num_classes)


    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        total_num = len(index_to_class)
        sample_num = round(total_num / client_per_class)

        cnt = 0
        prev_idx = 0
        for client_idx in range(client_num):
            if client_idx not in [(j + class_index) % client_num for j in range(0, client_num - client_per_class)]:
                cnt += 1
                if cnt != client_per_class:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:prev_idx + sample_num])
                    prev_idx += sample_num
                else:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:])

    for client_idx in range(client_num):
        np.random.shuffle(client_train_index[client_idx])

    targets_true = copy.deepcopy(data_train.targets)
    if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
        print(corruption_matrix)
        if corrupt_num == -1:
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)
        else:
            for client_idx in range(corrupt_num):
                for index in client_train_index[client_idx]:
                    p = corruption_matrix[int(data_train.targets[index])]
                    data_train.targets[index] = np.random.choice(num_classes, p=p)

    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([
                    data_train[i] for i in client_train_index[client_idx - 1]
                ], batch_size=args.batch_size, shuffle=False,
                collate_fn=None),
            'train_targets_true': [
                targets_true[i] for i in client_train_index[client_idx - 1]],
            'meta_train': [],
            'val':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    args.batch_size,
                    shuffle=False,
                collate_fn=None),
            'test':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    args.batch_size,
                    shuffle=False,
                collate_fn=None)
        }
        data_dict[client_idx - 1] = dataloader_dict

    return data_dict
