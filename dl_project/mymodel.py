import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

####################################################################
# creating transformer for DataLoader
####################################################################

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


##################################################################
# creating test and training DataLoaders
##################################################################

batch_size=100


train_loader = DataLoader(
        torchvision.datasets.MNIST(config['paths']['train_path'],    # for changing the path, change it in config.ini file
                            train=True, 
                            download=True, 
                            transform=transformer),
        batch_size, 
        shuffle=True)


test_loader = DataLoader(
        torchvision.datasets.MNIST(config['paths']['test_path'],     # for changing the path, change it in config.ini file
                            train=False,
                            download=True,
                            transform=transformer),
        batch_size,
        shuffle=True)



######################################################################
# creating model
######################################################################

input_size = 784
hidden_size1 = 128
hidden_size2 = 64
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_size1),
                      nn.ReLU(),
                      nn.Linear(hidden_size1, hidden_size2),
                      nn.ReLU(),
                      nn.Linear(hidden_size2, output_size))



##########################################################################
# loss function and optimizer
##########################################################################

loss_f = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=0.001)     # best result with lr=0.001



##########################################################################
# train the model
##########################################################################

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        images = inputs.view(inputs.shape[0], -1)
        optim.zero_grad()

        loss = loss_f(model(images), labels)
        loss.backward()
        optim.step()

        running_loss += loss.item()
    else:
        print("Epoch ", epoch, " - Training loss: ", running_loss/len(train_loader))



######################################################################################
# saving model
######################################################################################

torch.save(model.state_dict(), config['paths']['save_and_load_path'])          # for changing the path, change it in config.ini file




""" #######################################################################################
# loading model
########################################################################################

model.load_state_dict(torch.load(config['paths']['save_and_load_path']))         # for changing the path, change it in config.ini file
model.eval()


 """

#########################################################################################
# testing model with test_data
##########################################################################################


""" correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        test_images = images.view(images.shape[0], -1)

        outputs = model(test_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print("predicted: ", predicted)
        print("labels: ", labels)
        print("---------------------")


print('Accuracy of the network on test images: %d %%' % 
    (100 * correct / total)) """



#######################################################################
# function for testing my own data
#######################################################################

def testingmydata (my_loader, label):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in my_loader:
            images, labels = data, label
            test_images = images.view(images.shape[0], -1)

            outputs = model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return predicted



#################################################################################
# loading own images
#################################################################################

digits = [0,1,2,3,4,5,6,7,8,9]

for i in digits:
    my_data = Image.open("{}.png".format(i)) 
    my_data = transformer(my_data)

    my_loader = DataLoader(
        my_data,
        shuffle=True)
    label = torch.tensor([i])
    print("image of {}:".format(i))
    print("predicted:", testingmydata(my_loader, label))