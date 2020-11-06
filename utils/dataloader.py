# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for full license information.

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.utils import one_hot
from utils.transform import DefaultTransform, InitialCropPad, MinimalTargets, functional_fixed_size
from utils.labels import CITYSCAPE_LABELS, CARLA_LABELS, CGMU_LABELS


class Training_Dataset(Dataset):
    def __init__(self, data_dir, dataset, split, crop, fixed_size, device, labels2keep=None, transforms=None):
        """
        Initialize the dataset.
        :param data_dir: Path to the directory containing the RGB and Semantic images.
        :param dataset: Dataset to use (Carla or Cityscape).
        :param split: Dataset split to generate (Train, Valid or Test).
        :param crop: Size of the input (cropped or padded as necessary, centered).
        :param transforms: Transforms to apply on the dataset and labels as pre-processing.
        """

        self.default_transform = DefaultTransform(device)     # From Image to PIL to Tensor
        self.dataset = dataset
        self.split = split
        self.crop = crop
        self.fixed_size = fixed_size
        self.device = device
        self.labels2keep = labels2keep
        self.transforms = transforms

        if os.path.exists(data_dir):
            if self.split == "Train":
                self.image_dir = os.path.join(data_dir, 'RGB/train')
                self.label_dir = os.path.join(data_dir, 'Semantic/train')

            elif self.split == "Valid":
                self.image_dir = os.path.join(data_dir, 'RGB/val')
                self.label_dir = os.path.join(data_dir, 'Semantic/val')

            elif self.split == "Test":
                self.image_dir = os.path.join(data_dir, 'RGB/test')
                self.label_dir = os.path.join(data_dir, 'Semantic/test')

            else:
                print('Incorrect split in config. Train, Valid or Test')

        else:
            print('Incorrect data directory in config.')

        self.image_list = []
        self.label_list = []

        self.image_list = sorted(os.listdir(self.image_dir))
        self.label_list = sorted(os.listdir(self.label_dir))

        if self.dataset == "Cityscape":
            labels = CITYSCAPE_LABELS
        elif self.dataset == "Carla":
            labels = CARLA_LABELS
        elif self.dataset == "CGMU":
            labels = CGMU_LABELS
        else:
            print("Wrong dataset name in config.")
            assert 1 == 0

        if self.labels2keep:
            self.targetTransform = MinimalTargets(list_to_keep=labels2keep)
            self.labels = [labels[i] for i in labels2keep]

        else:
            self.labels = labels

        self.num_classes = len(self.labels)

        if self.fixed_size:
            self.functional_fixed_size = functional_fixed_size(fixed_size)

        if self.crop:
            self.initialCropPad = InitialCropPad(crop)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if index < len(self.image_list):
            image_path = os.path.join(self.image_dir, self.image_list[index])
            img_x = np.array(Image.open(image_path))[:, :, 0:3]
            
            label_path = os.path.join(self.label_dir, self.label_list[index])

            if self.dataset == "Cityscape":
                target = np.array(Image.open(label_path).convert('L'), dtype='long')
                target[target == -1] = 34
            elif self.dataset == "Carla":
                target = np.array(Image.open(label_path), dtype='long')[:, :, 0]
            elif self.dataset == "CGMU":
                target = np.array(Image.open(label_path), dtype='long')[:, :, 0]

            if self.labels2keep:
                target = self.targetTransform(target)

            initial_size = img_x.shape[0:2]

            img_x = self.default_transform(img_x)
            target = one_hot(target, self.num_classes, self.device)

            if self.fixed_size:
                img_x, target = self.functional_fixed_size((img_x, target))

            if self.crop:
                img_x, target = self.initialCropPad(img_x, initial_size, target)

            if self.transforms is not None:
                img_x, target = self.transforms((img_x, target))
            
            sample = {'image': img_x, 'target': target, 'initial_size': initial_size}

        else:
            print('Index out of Data Range')

        return sample


class Extra_Dataset(Dataset):
    def __init__(self, data_dir, crop, fixed_size, device, transforms=None):
        """
        Initialize the dataset.
        :param data_dir: Path to the directory containing the RGB images.
        :param crop: Size of the input (cropped or padded as necessary, centered).
        :param transforms: Transforms to apply on the dataset and labels as pre-processing.
        """

        self.default_transform = DefaultTransform(device)     # From Image to PIL to Tensor
        self.fixed_size = fixed_size
        self.crop = crop
        self.transforms = transforms

        if os.path.exists(data_dir):
            self.image_dir = os.path.join(data_dir, 'RGB/extra')

        else:
            print('Incorrect data directory in config.')

        self.image_list = os.listdir(self.image_dir)
        
        if self.fixed_size:
            self.functional_fixed_size = functional_fixed_size(fixed_size)

        if self.crop:
            self.initialCropPad = InitialCropPad(crop)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if index < len(self.image_list):
            image_path = os.path.join(self.image_dir, self.image_list[index])
            img_x = np.array(Image.open(image_path))[:, :, 0:3]

            initial_size = img_x.shape[0:2]

            img_x = self.default_transform(img_x)

            if self.fixed_size:
                img_x = self.functional_fixed_size(img_x)
          
            if self.crop:
                img_x = self.initialCropPad(img_x, initial_size)

            if self.transforms is not None:
                img_x = self.transforms(img_x)
            
            sample = {'image': img_x, 'initial_size': initial_size}

        else:
            print('Index out of Data Range')

        return sample


class Predict_Dataset(Dataset):
    def __init__(self, data_dir, dataset, prediction_list_file,
                 crop, fixed_size, device, labels2keep=None, transforms=None):
        """
        Initialize the prediction dataset.
        :param data_dir: Path to the directory containing the RGB and Semantic images.
        :param dataset: Dataset to use (Carla or Cityscape).
        :param prediction_list_file: List of images to predict.
        :param crop: Size of the input (cropped or padded as necessary, centered).
        :param transforms: Transforms to apply on the dataset and labels as pre-processing.
        """

        self.default_transform = DefaultTransform(device)     # From Image to PIL to Tensor
        self.dataset = dataset
        self.fixed_size = fixed_size
        self.crop = crop
        self.labels2keep = labels2keep
        self.transforms = transforms

        if os.path.exists(data_dir):
            self.data_dir = os.path.join(data_dir, "RGB")

        else:
            print('Incorrect data directory in config.')

        self.image_list = []

        if os.path.exists(prediction_list_file):
            self.image_list = np.loadtxt(prediction_list_file, str)
        
        if self.dataset == "Cityscape":
            labels = CITYSCAPE_LABELS
        elif self.dataset == "Carla":
            labels = CARLA_LABELS
        elif self.dataset == "CGMU":
            labels = CGMU_LABELS
        else:
            assert 1 == 0

        if self.labels2keep:
            self.labels = [labels[i] for i in labels2keep]

        else:
            self.labels = labels

        self.num_classes = len(self.labels)

        if self.fixed_size:
            self.functional_fixed_size = functional_fixed_size(fixed_size)

        if self.crop:
            self.initialCropPad = InitialCropPad(crop)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if index < len(self.image_list):
            image_path = os.path.join(self.data_dir, self.image_list[index])
            img_x = np.array(Image.open(image_path))[:, :, 0:3]

            initial_size = img_x.shape[0:2]

            img_x = self.default_transform(img_x)

            if self.fixed_size:
                img_x = self.functional_fixed_size(img_x)

            if self.crop:
                img_x = self.initialCropPad(img_x, initial_size)

            if self.transforms is not None:
                img_x = self.transforms((img_x))

            sample = {'image': img_x, 'name': self.image_list[index].split('/')[1], 'initial_size': initial_size}

        else:
            print('Index out of Data Range')

        return sample


def custom_dataloaders(config, device, extra=None, transforms=None):
    try:
        _ = config['dataset']['fixed_size']
    except KeyError:
        config['dataset']['fixed_size'] = None

    train_set = Training_Dataset(data_dir=config['dataset']['data_dir'],
                                 dataset=config['dataset']['name'],
                                 split='Train',
                                 crop=config['dataset']['crop'],
                                 fixed_size=config['dataset']['fixed_size'],
                                 device=device,
                                 labels2keep=config['dataset']['labels2keep'],
                                 transforms=transforms)

    valid_set = Training_Dataset(data_dir=config['dataset']['data_dir'],
                                 dataset=config['dataset']['name'],
                                 split='Valid',
                                 crop=config['dataset']['crop'],
                                 fixed_size=config['dataset']['fixed_size'],
                                 device=device,
                                 labels2keep=config['dataset']['labels2keep'],
                                 transforms=transforms)

    test_set = Training_Dataset(data_dir=config['dataset']['data_dir'],
                                dataset=config['dataset']['name'],
                                split='Test',
                                crop=config['dataset']['crop'],
                                fixed_size=config['dataset']['fixed_size'],
                                device=device,
                                labels2keep=config['dataset']['labels2keep'],
                                transforms=transforms)

    predict_set = Predict_Dataset(data_dir=config['dataset']['data_dir'],
                                  dataset=config['dataset']['name'],
                                  prediction_list_file=config['dataset']['prediction_list'],
                                  crop=config['dataset']['crop'],
                                  fixed_size=config['dataset']['fixed_size'],
                                  device=device,
                                  labels2keep=config['dataset']['labels2keep'],
                                  transforms=transforms)
                            
    train_loader = DataLoader(
        train_set,
        batch_size=config['dataloader']['train']['batch_size'],
        shuffle=config['dataloader']['train']['shuffle'])

    valid_loader = DataLoader(
        valid_set,
        batch_size=config['dataloader']['valid']['batch_size'],
        shuffle=config['dataloader']['valid']['shuffle'])

    test_loader = DataLoader(
        test_set,
        batch_size=config['dataloader']['test']['batch_size'],
        shuffle=config['dataloader']['test']['shuffle'])

    predict_loader = DataLoader(
        predict_set,
        batch_size=config['dataloader']['prediction']['batch_size'],
        shuffle=config['dataloader']['prediction']['shuffle'])

    if extra:
        extra_set = Extra_Dataset(data_dir=config['dataset']['data_dir'],
                                  crop=config['dataset']['crop'],
                                  fixed_size=config['dataset']['fixed_size'],
                                  device=device,
                                  transforms=transforms)

        extra_loader = DataLoader(
            extra_set,
            batch_size=config['dataloader']['extra']['batch_size'],
            shuffle=config['dataloader']['extra']['shuffle'])

        return train_loader, valid_loader, test_loader, extra_loader, predict_loader

    else:
        return train_loader, valid_loader, test_loader, predict_loader


if __name__ == "__main__":
    pass
