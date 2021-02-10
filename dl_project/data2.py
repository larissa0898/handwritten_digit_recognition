from mnist import MNIST
import numpy as np
import torch
import torchvision
import random

num_workers = 0
batch_size = 20
valid_size = 0.2

transform = torchvision.transforms.ToTensor()

train_data = torchvision.datasets.MNIST(root='data', train=True, 
                                download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='data', train=False, 
                                download=True, transform=transform)



num_train = len(train_data)
indices = list(range(num_train))
random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()