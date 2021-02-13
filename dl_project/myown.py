import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


##################################################################
# creating DataLoaders
##################################################################
batch_size=100
train_path = 'C:\\Users\\laris\\Desktop\\dl_project\\data'
test_path = 'C:\\Users\\laris\\Desktop\\dl_project\\data'

train_loader = DataLoader(
        torchvision.datasets.MNIST(train_path, 
                            train=True, 
                            download=True, 
                            transform=transformer),
        batch_size, 
        shuffle=True)


test_loader = DataLoader(
        torchvision.datasets.MNIST(test_path,
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
# training the model
##########################################################################

epoch = 10

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

#torch.save(model, 'C:\\Users\\laris\\Desktop\\GitHub\\deep_learning_project\\dl_project')




#######################################################################################
# loading model
########################################################################################

#model = torch.load('C:\\Users\\laris\\Desktop\\GitHub\\deep_learning_project\\dl_project')
model.eval()



#########################################################################################
# testing model with test_data
##########################################################################################


#dataiter = iter(test_loader)
#images, labels = dataiter.next()

#test_images = images.view(images.shape[0], -1)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        test_images = images.view(images.shape[0], -1)

        outputs = model(test_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on 10000 test images: %d %%' % 
    (100 * correct / total))