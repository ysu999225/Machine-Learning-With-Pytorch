import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!
        #Define ALL of the sub-modules in the order that the input data goes through.
        #DO NOT change the order of sub-modules
        #Define ALL the sub-modules with an instance of torch.nn.Module, e.g., torch.nn.Conv2d.

        # Please ONLY define the sub-modules here
        
        #A 2D convolutional layer (torch.nn.Conv2d) with 7 output channels, with kernel size 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,7,3),
            nn.BatchNorm2d(7),
            nn.ReLU()
        )
        #A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, 
        # with kernel size 3. Please use padding=1 for this layer.
        self.layer2 = nn.Sequential(
            nn.Conv2d(7,3,3,padding = 1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        #A 2D maximimum pooling layer (torch.nn.MaxPool2d), with kernel size 2
        self.layer3 = torch.nn.MaxPool2d(2)
        #A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, with kernel size 2
        self.layer4 = nn.Sequential(
            nn.Conv2d(3,3,2),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        #A fully connected (torch.nn.Linear) layer with 10 output features
        self.fc = torch.nn.Linear(12,10)
        
        

        
        

        

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        #Convert 3D to 4D
        xb = xb.unsqueeze(1)
        
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = self.layer3(xb)
        xb = self.layer4(xb)
        
        xb = xb.reshape(xb.size(0),-1)
        
        xb = self.fc(xb)
        
        return xb
        
        

class DigitsConvNetv2(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        You can customize your network structure here as long as the input shape and output shape are as specified.

        '''
        super(DigitsConvNetv2, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.layer2 = nn.Sequential(
            #set the stride = 1
            nn.Conv2d(16,32,3,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.layer3 = torch.nn.MaxPool2d(2)
        

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
    
        self.fc1 = torch.nn.Linear(256,128)
        self.fc2 = torch.nn.Linear(128,10)
        
     
        
        
 
        

        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        #Convert 3D to 4D
        xb = xb.unsqueeze(1)
        
        xb = self.layer1(xb)
        xb = self.layer2(xb)
        xb = self.layer3(xb)
        xb = self.layer4(xb)
        
        xb = xb.reshape(xb.size(0),-1)
        
        xb = self.fc1(xb)
        xb = self.fc2(xb)
        
        return xb

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    
    # Train the network for n_epochs, storing the training and validation losses
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)
    #add 'net.eval()' before computing the losses function
    net.eval()
    # being sure not to store gradient information (e.g. with torch.no_grad():)
    # Compute the loss on the training and validation sets at the start,
    with torch.no_grad():
        train_losses = []
        test_losses = []
    # Training the batches of data
    for epoch in range(n_epochs):
        net.train()
        # Training the batches of data
        for x, y in train_dl:
            # Forward pass
            predction = net(x)
            # Loss calculation
            loss = loss_func(predction, y)
            # Backpropa
            loss.backward()
            # Update the weights
            optimizer.step()
            # Zero gradients
            optimizer.zero_grad()
    
       # after every epoch. Remember not to store gradient information while calling 
        net.eval()
        with torch.no_grad():
            train_losses = []
            test_losses = []
        
    return train_losses, test_losses