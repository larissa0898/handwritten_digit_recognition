import torch
import torch.nn as nn
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Neural Network Layer 
        self.convolutational_neural_network_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2)
        )

        # Linear layer
        self.linear_layers = nn.Sequential(
                nn.Linear(in_features=24*7*7, out_features=64),          
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=64, out_features=10)
        )

    # Defining the forward pass 
    def forward(self, x):
        x = self.convolutational_neural_network_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x



def train_model(model, epochs, optimizer, train_loader, loss_func):
    """ Function for training a model.
     """

    for epoch in range(epochs):
   
        total_train_loss = 0
        total = 0
        model.train()

        for idx, (image, label) in enumerate(train_loader):
            image, label = image, label

            optimizer.zero_grad()

            output = model(image)

            loss = loss_func(output, label)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            for i, predicted in enumerate(output):
                if label[i] == torch.max(predicted.data, 0)[1]:
                    total = total + 1
    torch.save(model.state_dict(), config['paths']['save_and_load_path'])



def testingmydata (my_loader, model):
    with torch.no_grad():
        for data in my_loader:
            images = data.view(1,1,28,28)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
    return predicted