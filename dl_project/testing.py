import click
import os
import cv2 as cv2
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from configparser import ConfigParser
from model import Net, train_model, testingmydata
from preprocessingfunctions import firstpreprocessing, secondpreprocessing, show_train_images
import matplotlib.pyplot as plt
import numpy as np


config = ConfigParser()
config.read('config.ini')


batch_size = 100

######################################################
# transformer for DataLoader
###################################################
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                                ])

###############################################################################
# DataLoaders for MNIST dataset
###############################################################################
train_loader = DataLoader(
        datasets.MNIST(config['paths']['train_path'],      # for changing the path, change it in config.ini file
                            train=True, 
                            download=True, 
                            transform=transform),    
        batch_size, 
        shuffle=True)


test_loader = DataLoader(
        datasets.MNIST(config['paths']['test_path'],     # for changing the path, change it in config.ini file
                            train=False,
                            download=True,
                            transform=transform),
        batch_size,
        shuffle=True)


###########################################################################
# Do you want to see some of the MNIST images? 
# Y - 6 images will be shown
# n - programm runs further
########################################################################### 

if click.confirm('Do you want to see some of the MNIST images?', default=True):
    show_train_images(train_loader)


###################################################################################
# create model, optimizer and loss function object and define the epochs
##################################################################################

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.02)
loss_func = nn.CrossEntropyLoss()
epochs = 20


###########################################################################
# Do you want to train a new model? 
# Y - training function is called
# n - old model will be loaded into script
########################################################################### 

if click.confirm('Do you want to train a new model?', default=True):
    train_model(model, epochs, optimizer, train_loader, loss_func)      # training takes about 8 minutes


##########################################################################
# load trained model
##########################################################################

model.load_state_dict(torch.load(config['paths']['save_and_load_path']))    # for changing the path, change it in config.ini file


##########################################################
# test model with MNIST test set
##########################################################
model.eval()
total = 0
correct = 0
for i, (image, label) in enumerate(test_loader):
    image, label = image, label
    output = model(image)                              # applies test images to the model
    loss = loss_func(output, label)

    for j, predicted in enumerate(output):            # correctly recognized images and total images are summed up to get accuracy
        if label[j] == torch.max(predicted.data, 0)[1]:    
            correct += 1
        total += 1

print("\nACCURACY OF MNIST TEST IMAGES: {}%\n".format((correct / total)*100))

#################################################################################
# loading own images and pre-process them
#################################################################################
digits = [0,1,2,3,4,5,6,7,8,9]

correct = 0
total = 0
print("Own input images and corresponding prediction: \n")

for root, subdirectories, files in os.walk(config['image']['images']):           # iterates over all subdirectories and opens every image in each subdirectory
    for subdirectory in subdirectories:                                          # to apply it to the model (total of 60 images)
        folder = os.path.join(root, subdirectory)
        for i in digits:
            image = cv2.imread(os.path.join(folder,"initial{}.jpg".format(i)))   # 
            my_data = firstpreprocessing(image)                # if you want another pre-processing, simply replace 'firstpreprocessing' with 'secondpreprocessing'
            my_data = transform(my_data)                       # transformer from line 25

            my_loader = DataLoader(                            # own data is loaded with DataLoader
                my_data)
    
            label = torch.tensor([i])                          # images are arranged in a way, to open in order from 0 to 9, so labels always correspond to the current index with which images are opened (line 120)
            predicted = testingmydata(my_loader, model)        # testingmydata-Function in 'model.py' is called

            print("image of {}:".format(i))                    # prints the number, which is represented in the image
            print("predicted: {}\n".format(predicted.item()))  # prints the predicted digit
    
            correct += (predicted == label).sum()
            total += 1

print("ACCURACY OF OWN IMAGES: {}%".format(float(correct)/total*100))