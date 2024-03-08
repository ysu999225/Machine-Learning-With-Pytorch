import torch
import hw2_utils 
import matplotlib.pyplot as plt


'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    # number of examples and features
    n, d = X.size()
    #initialize the weights w
    # the number of parameters: d + 1, because we will add the bias term later
    w = torch.zeros(d + 1, 1)
    #create the bias term as a column of ones
    ones = torch.ones(n,1)
    #add the ones to the matrix X as the bias term
    #torch.cat function helps combine multiple tensors into one dimension
    bias = torch.cat((ones,X),dim = 1)
  
    
    for i in range(num_iter):
        # for the current parameters, get the difference between the predictions and true
        difference = torch.matmul(bias,w) - Y
        # multiplication of matrix and difference and across all examples
        # the mean square error
        gradients = torch.sum(bias * difference, dim = 0) * 2 / n
        #reshape the gradient
        gradients = torch.reshape(gradients, (-1, 1))
         # after this process, the updated weight should be
         #consider the learning rate
        w = w - lrate * gradients
       
    return w
        


def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    # number of examples and features
    n, d = X.size()
    #initialize the weights w
    # the number of parameters: d + 1, because we will add the bias term later
    w = torch.zeros(d + 1, 1)
    #create the bias term as a column of ones
    ones = torch.ones(n,1)
    #add the ones to the matrix X as the bias term
    #torch.cat function helps combine multiple tensors into one dimension
    bias = torch.cat((ones,X),dim = 1)
    # consider the torch.pinverse to get the inverse of the matrix
    inverse = torch.pinverse(torch.matmul(bias.t(),bias))
    # the update w should be
    w = torch.matmul(torch.matmul(inverse,bias.t()),Y)
    
    return w

    


def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''

    #load the data
    X,Y = hw2_utils.load_reg_data()
    #use linear_norm() to calculate the regression results w
    w = linear_normal(X, Y)
    #using the weights
    n = X.size()[0]
    ones = torch.ones(n,1)
    bias = torch.cat((ones,X),dim = 1)
    #get the predicitions
    prediction = torch.matmul(bias,w)
    #plot the points of the dataset
    plt.scatter(X.numpy(),Y.numpy(),label = 'the points of the dataset',color = 'Red')
    #plot the regression curve
    plt.plot(X.numpy(),prediction.numpy(),label = 'Regression curve')
   
    plt.legend()
    plt.title('Linear Regression Plot')
    plt.xlabel('X_Values')
    plt.ylabel('Y_Values')
    
    # Return the figure object
    return plt.gcf()

# Test the function
fig = plot_linear()
plt.show()
    
    
    
    
    
    


# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    # almost the same as above
    
    # number of examples and features
    n, d = X.size()
    #initialize the weights w
    # the number of parameters: d + 1, because we will add the bias term later
    w = torch.zeros(d + 1, 1)
    #create the bias term as a column of ones
    ones = torch.ones(n,1)
    #add the ones to the matrix X as the bias term
    #torch.cat function helps combine multiple tensors into one dimension
    bias = torch.cat((ones,X),dim = 1)
    
    for i in range(num_iter):
        # compute the weighted inputs
        input = torch.matmul(bias,w)
        #calculate the exp term follow the formula
        exp = torch.exp(-Y * input)
        # get the numerator first
        numerator = exp * (-Y) * bias
        # get the denominator
        denominator = 2 + exp
        # get the gradient
        gradients = torch.sum (numerator / denominator, dim = 0) * 1 / n
        #reshape the gradients, consider the bias term
        gradients = torch.reshape(gradients,(-1, 1))
        # after this process, the updated weight should be
        #consider the learning rate
        w = w - lrate * gradients
       
    return w
    


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    

    #load the data
    X,Y = hw2_utils.load_logistic_data()
    #use logistic(X,Y) to calculate the regression results w
    log_w= logistic(X, Y)
    #also use n linear gd(X,Y)
    linear_w = linear_gd(X,Y)
    #using the weights
    n = X.size()[0]
    ones = torch.ones(n,1)
    bias = torch.cat((ones,X),dim = 1)
    #get the predicitions of each formula
    #based on the formula, we got the logistic regression prediction first 
    logs = torch.exp(torch.matmul(bias, log_w))
    p_log = 1 / (1 + logs)
    # get the linear regression prediction second
    p_linear =  torch.matmul(bias, linear_w)
    # get the boundary
    boundary = torch.linspace(torch.min(X[:,0]), torch.max(X[:,1]),30)

    x2_log = -(log_w[0]+log_w[1]*boundary)/log_w[2]
    x2_lin = -(linear_w[0] + linear_w[1] * boundary) / linear_w[2]
  

    plt.scatter(X[:,0], X[:,1], c = Y)
    plt.plot(boundary, x2_log, label='Logistic ',color = 'Red')
    plt.plot(boundary, x2_lin, label='Linear_GD', color = 'Blue')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 1)
    plt.show()
    return plt.gcf()

# Test the function
fignew = logistic_vs_ols()
plt.show()

