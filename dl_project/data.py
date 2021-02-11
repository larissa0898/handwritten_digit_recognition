import numpy as np
import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import idx2numpy
import matplotlib.pyplot as plt
import gzip

####################################################
# to print images
###################################################
#file1 = 'train-images.idx3-ubyte'
#imagearray = idx2numpy.convert_from_file(file1)

#plt.imshow(imagearray[1], cmap=plt.cm.binary)
#plt.show()
#########################################################

#########################################################################
# images into numpy array
#########################################################################

def training_images():
    with gzip.open('train-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images

#print(type(training_images()))             <---- return type: numpy array


######################################################################
# make labels visible and into numpy array
#####################################################################

def training_labels():
    with gzip.open('train-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

labels_tensor = torch.from_numpy(training_labels())                       # numpy-array of labels into torch.tensor of labels


#########################################################################################
# one-hot_encoding of labels
#########################################################################################
x = torch.nn.functional.one_hot(labels_tensor.long(), num_classes=10)
result = torch.reshape(x, (600,100,10))

#stacked_tensor = torch.stack(listoftensors)                                   # falls alle Labels in einen einzigen Tensor rein sollen und nicht in
#stacked_tensor = torch.reshape(stacked_tensor, (60000,10))                    # 600 verschiedenen




""" 
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


###############################################################
# One-Hot_Encoding for labels
###############################################################

listoftensors = []

for batch_size, (_, y) in enumerate(train_loader):
        x = torch.nn.functional.one_hot(y, num_classes=10)
        listoftensors.append(x)


#stacked_tensor = torch.stack(listoftensors)                                   # falls alle Labels in einen einzigen Tensor rein sollen und nicht in
#stacked_tensor = torch.reshape(stacked_tensor, (60000,10))                    # 600 verschiedenen
 """