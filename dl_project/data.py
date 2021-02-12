import numpy as np
import gzip
import torch
import torch.nn as nn
#import torchvision
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader

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


new_images = []

for image in training_images():                                    # image pixel from 0-255 to 0.0-1.0
    for pixel in image:
        new_images.append(pixel/255.0)

new_images = np.array(new_images)                                 # new_images von Liste in numpy array umwandeln

image_tensor = torch.from_numpy(new_images)                       # numpy array zu Tensor umwandeln
image_tensor = torch.reshape(image_tensor, (60000, 28, 28))       # Tensor in richtige Form bringen


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
labels_tensor = labels_tensor.long()                                      # into long values for one-hot encoding

###########################################################################################
# create new data set with tuples; Liste mit Tupeln drin [(image1, label1), (image2, label2),...]
###########################################################################################
data = []

for i in range(len(training_labels())):
    data.append((image_tensor[i], labels_tensor[i]))


#############################################################################################
# image encoding
############################################################################################


# ?



#########################################################################################
# one-hot_encoding of labels and label embedding
#########################################################################################
enc_data = []
E = torch.randn(10)

for image, label in data:
    enc_label = torch.nn.functional.one_hot(label, num_classes=10)
    enc_label = enc_label.float()*E
    enc_data.append((image, enc_label))


#stacked_tensor = torch.stack(listoftensors)                                   # falls alle Labels in einen einzigen Tensor rein sollen und nicht in
#stacked_tensor = torch.reshape(stacked_tensor, (60000,10))                    # 600 verschiedenen

"""
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# DataLoaders
batch_size=100
train_set = 'C:\\Users\\laris\\Desktop\\dl_project\\data'
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


########################################################
# create model
########################################################

#model = nn.Sequential(nn.Linear(input_size, hidden_size1),
                       # nn.ReLU()
                        #nn.Linear(hidden_size1, hidden_size2),
                        #nn.ReLU(),
                        #nn.Linear(hidden_size2, output_size),
                        #nn.LogSoftmax(dim=1)
                        #)