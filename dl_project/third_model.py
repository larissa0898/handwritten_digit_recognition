import click
import cv2 as cv2
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from configparser import ConfigParser
from model_zu_third import Net, train_model, testingmydata
from preprocessingfunctions import firstpreprocessing, secondpreprocessing



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

###############################################################################
# DataLoaders
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

###################################################################################
# create model, opitmizer and loss function object
##################################################################################

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()


###########################################################################
# Do you want to train a new model? 
# Y - training function is called
# n - old model will be loaded into script
########################################################################### 

if click.confirm('Do you want to train a new model?', default=True):
    train_model(model, 20, optimizer, train_loader, loss_func)


##########################################################################
# load trained model
##########################################################################

model.load_state_dict(torch.load(config['paths']['save_and_load_path']))


##########################################################
# test model with MNIST test set
##########################################################
model.eval()
total = 0
correct = 0
for i, (image, label) in enumerate(test_loader):
    image, label = image, label
    output = model(image)
    loss = loss_func(output, label)


    for j, predicted in enumerate(output):
        if label[j] == torch.max(predicted.data, 0)[1]:
            correct += 1
        total += 1

print("Accuracy of test images: ", (correct / total)*100, "%")



#################################################################################
# loading own images and pre-process them
#################################################################################
digits = [0,1,2,3,4,5,6,7,8,9]

correct = 0
total = 0
for i in digits:
    image = cv2.imread("initial{}.jpg".format(i))
    my_data = firstpreprocessing(image)                # if you want another preprocessing, then simply replace 'firstpreprocessing' with 'secondpreprocessing'
    my_data = transform(my_data)

    my_loader = DataLoader(
        my_data)
    
    label = torch.tensor([i])
    predicted = testingmydata(my_loader, model)


    print("image of {}:".format(i))
    print("predicted:", predicted)
    
    correct += (predicted == label).sum()
    total += 1

print(float(correct)/total*100)