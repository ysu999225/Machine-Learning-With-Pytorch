import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def linear_kernel(x, y):
    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
    
    Returns:
        a torch.float32 scalar
    '''
    # with torch.no_grad will return two vectors x and y without keeping track of gradients
    # torch.no_grad() not use the gradient descent
    with torch.no_grad():
        # need to compute the dot product of the two vectors x and y
        return torch.dot(x,y)
        

def polynomial_kernel(x, y, p):
    '''
    Compute the polynomial kernel function with arbitrary power p

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        p: the power of the polynomial kernel
    
    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        #based on the formula K(x,y) = (xTy + c) ^ p
        # from the lecture slides, we set the c = 1
        return(torch.dot(x,y)+1) ** p

def gaussian_kernel(x, y, sigma):
    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        sigma: parameter sigma in rbf kernel
    
    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        # based on the formula from the Gaussian kernel
        #first get the L2 distance between x and y
        distance = torch.sum((x-y)**2)
        # 2 sigma ^ 2
        return torch.exp(-distance / (2 * sigma ** 2))

def svm_epoch_loss(alpha, x_train, y_train, kernel=linear_kernel):
    '''
    Compute the linear kernel function

    Arguments:
        alpha: 1d tensor with shape (N,), alpha is the trainable parameter in our svm 
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
    
    Returns:
        a torch.float32 scalar which is the loss function of current epoch
    '''
    # the number of dual cofficients 
    N = alpha.shape[0]
    #initialize the doublesum to 0, store the value in the loop
    # torch.float32 scalar
    doublesum = 0.0
    for i in range(N):
        for j in range(N):
            #the part of the SVM dual objective function
            doublesum += alpha[i] * alpha[j] * y_train[i] * y_train[j] * kernel(x_train[i],x_train[j])
    # the loss should calculate out the loop, based ont the loss function
    # negative loss from campuswire
    #the loss function often returns the negative of the dual objective so that a minimization algorithm
    loss = 0.5 * doublesum - torch.sum(alpha)
    return loss
    

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=linear_kernel, c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is the linear kernel.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    ''' 
    # the number of training data samples
    #x_train (N,d)
    N = x_train.shape[0]
    #Initialize alpha to be 0.
    #requires_grad = True
    #indicates that the tensor should track operations performed on it, so that gradients with respect to that tensor can be computed during backpropagation
    alpha = torch.zeros(N,requires_grad = True)
    # using the svm_epoch_loss function
    for _ in range(num_iters):
        loss = svm_epoch_loss(alpha, x_train, y_train, kernel)
    # use backward function
        loss.backward()
    # also consider torch.no_grad() not use the gradient descent
        with torch.no_grad():
        #update the Gradient Descent alpha
            alpha -= lr * alpha.grad
        # consider the alpha = alpha.clamp(...）
        #Test Failed: clamp() received an invalid combination of arguments - got (Tensor, max=int, min=float), but expected one of:
        # (Tensor min, Tensor max)
        #(Number min, Number max)
            #alpha.clamp(min = 0.0, max = c)
            if c is None:
                # we need to use the clamp_here to avoid The distance between your primal solution and the optimal is too large: inf
                alpha.clamp_(min = 0.0)
            else:
                alpha.clamp_(min = 0.0, max = c)
        # then still keep the updated alpha requires_grad = True
            alpha.requires_grad = True
        # set the zero for the gradient for the next loop
        #Test Failed: 'Tensor' object has no attribute 'zero'
        # - This uses an in-place operation. The trailing underscore (`_`) in PyTorch denotes that the operation is performed in-place on the tensor. 
        # This means the tensor's memory is directly modified and no new tensor is returned.
        # if use alpha.grad.zero(), It would return a new tensor with the same shape as `alpha.grad` where all values are zero
            alpha.grad.zero_()
        #Return alpha.detach() could possibly help you save some time
    return alpha.detach()
        
    
        
    
    
    

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=linear_kernel):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    idx = torch.nonzero(alpha,as_tuple=True)
    alpha_ = alpha[idx]
    x_ = x_train[idx]
    y_ = y_train[idx]
    if len(alpha_) == 0:
        return torch.zeros((x_test.shape[0],))
    id = alpha_.argmin()
    b = 1/y_[id]
    for i in range(len(alpha_)):
        b -= alpha_[i]*y_[i]*kernel(x_[i],x_[id])
    y_test = torch.zeros((x_test.shape[0],))
    for j in range(len(x_test)):
        y_test[j] = b
        for i in range(len(alpha_)):
            y_test[j] += alpha_[i]*y_[i]*kernel(x_[i],x_test[j])
    return y_test

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



