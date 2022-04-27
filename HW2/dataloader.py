import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from view_generator import ContrastiveLearningViewGenerator
from PIL import Image
import glob
import os

def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomRotation(10),
                                          transforms.RandomResizedCrop(size=48),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    return data_transforms


class ImageDataset(Dataset):
    def __init__(self, path, n_views):
        self.root_path = path
        self.file_name = glob.glob(self.root_path+"/*")
        self.file_name.sort()
        self.n_views = n_views
        
        self.transformation = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(size=96), self.n_views)

    def __getitem__(self, index):
        img_name = self.file_name[index]
        img = Image.open(img_name)
        img = self.transformation(img)
        return img

    def __len__(self):
        return len(self.file_name)

class evaluatedata():
    def __init__(self, root_path="./hw2/test"):
        self.root_path = root_path
    def get_data(self):
        img_data = ImageFolder(self.root_path, transform=transforms.ToTensor())
        return img_data

class ImageNPY(Dataset):
    def __init__(self, path="./hw2/unlabeled"):
        self.root_path = path
        self.file_name = glob.glob(self.root_path+"/*")
        self.file_name.sort()
        self.transformation = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img_name = self.file_name[index]
        img = Image.open(img_name)
        img = self.transformation(img)
        return img

    def __len__(self):
        return len(self.file_name)

