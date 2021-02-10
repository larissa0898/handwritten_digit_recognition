import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# DataLoaders
batch_size=100
train_path = 'C:\\Users\\laris\\Desktop\\dl_project\\data'
test_path = 'C:\\Users\\laris\\Desktop\\dl_project\\data'

train_loader = DataLoader(
        torchvision.datasets.MNIST(train_path, 
                            train=True, 
                            download=True, 
                            transform=transformer),
        batch_size, 
        shuffle=True
)

test_loader = DataLoader(
        torchvision.datasets.MNIST(test_path,
                            train=False,
                            download=True,
                            transform=transformer),
        batch_size,
        shuffle=True
)
