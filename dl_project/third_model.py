import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from configparser import ConfigParser
from model_zu_third import Network
from PIL import Image
from preprocessingfunctions import secondpreprocessing


config = ConfigParser()
config.read('config.ini')

batch_size = 64
######################################################
# transformer for DataLoader
###################################################
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                                ])

#####################################################################################
# data sets train and test
####################################################################################
train_set = torchvision.datasets.MNIST(config['paths']['train_path'],
                            train=True, 
                            download=True, 
                            transform=transform)

test_set = torchvision.datasets.MNIST(config['paths']['test_path'],     # for changing the path, change it in config.ini file
                            train=False,
                            download=True,
                            transform=transform)

###############################################################################
# DataLoaders
###############################################################################
train_loader = DataLoader(
        train_set,    
        batch_size, 
        shuffle=True)


test_loader = DataLoader(
        test_set,
        batch_size,
        shuffle=True)



model = Network()


optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()



epochs = 20
train_loss, val_loss = [], []
accuracy_total_train, accuracy_total_val = [], []

""" for epoch in range(epochs):
   
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    
    total = 0
    # training our model
    for idx, (image, label) in enumerate(train_loader):

        image, label = image, label

        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1
                
    accuracy_train = total / len(train_set)
    accuracy_total_train.append(accuracy_train)

    total_train_loss = total_train_loss / (idx + 1)
    train_loss.append(total_train_loss)

torch.save(model.state_dict(), config['paths']['save_and_load_path']) """


model.load_state_dict(torch.load(config['paths']['save_and_load_path']))

##########################################################
model.eval()
total = 0
for idx, (image, label) in enumerate(test_loader):
    image, label = image, label
    pred = model(image)
    loss = criterion(pred, label)
    #total_val_loss += loss.item()

    pred = torch.nn.functional.softmax(pred, dim=1)
    for i, p in enumerate(pred):
        if label[i] == torch.max(p.data, 0)[1]:
            total = total + 1

    accuracy_val = total / len(test_set)
    #print("accuracy test: ", accuracy_val)

#total_val_loss = total_val_loss / (idx + 1)
#val_loss.append(total_val_loss)
""" 
    if epoch % 5 == 0:
        print("Epoch: {}/{}  ".format(epoch, epochs),
            "Training loss: {:.4f}  ".format(total_train_loss),
            "Testing loss: {:.4f}  ".format(total_val_loss),
            "Train accuracy: {:.4f}  ".format(accuracy_train),
            "Test accuracy: {:.4f}  ".format(accuracy_val)) """

#######################################################################
# function for testing my own data
#######################################################################

def testingmydata (my_loader):
    with torch.no_grad():
        for data in my_loader:
            images = data.view(1,1,28,28)
            #test_images = images.view(1,1,28,28)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
    return predicted


################################################################################
# pre-processing my images
################################################################################


""" for i in digits:
    image = Image.open("initial{}.jpg".format(i))
    img = secondpreprocessing(image)
    img.save("{}.jpg".format(i)) """



#################################################################################
# loading own images and pre-processing them
#################################################################################
digits = [0,1,2,3,4,5,6,7,8,9]

correct = 0
total = 0
for i in digits:
    image = Image.open("initial{}.jpg".format(i))
    my_data = secondpreprocessing(image)
    my_data = transform(my_data)

    my_loader = DataLoader(
        my_data)
    
    label = torch.tensor([i])
    predicted = testingmydata(my_loader)


    print("image of {}:".format(i))
    print("predicted:", predicted)
    
    correct += (predicted == label).sum()
    total += 1

print(float(correct)/total*100)