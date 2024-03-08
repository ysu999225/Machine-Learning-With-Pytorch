import hw1_utils as utils
# choose the library you want to use
#import numpy as np
import torch
#import scipy
#import scipy.spatial
#import matplotlib.pyplot as plt
#from sklearn.datasets import load_iris


# Problem Naive Bayes
def bayes_MAP(X, y):
    '''
    Arguments:
        X (S x N LongTensor / Numpy ndarray): features of each object, X[i][j] = 0/1
        y (S LongTensor  / Numpy ndarray): label of each object, y[i] = 0/1

    Returns:
        D (2 x N Float Tensor / Numpy ndarray): MAP estimation of P(X_j=1|Y=i)

    '''

    S = X.shape[0] # number of objects
    N = X.shape[1] # number of features
    result = torch.zeros((2, N))
    # objects in each calss
    m = torch.sum(y == 0) #number of instance y = 0
    n = torch.sum(y == 1)# number of instance y = 1
    # the probabilities in features of each object
    for j in range (N): # for each j
        result[0, j] = torch.sum((X[:, j] == 1) & (y == 0)) / m #P(Xj=1∣Y=0)
        result[1, j] = torch.sum((X[:, j] == 1) & (y == 1)) / n #P(Xj=1∣Y=1)

    return result
    

def bayes_MLE(y):
    '''
    Arguments:
        y (S LongTensor / Numpy ndarray): label of each object

    Returns:
        p (float or scalar Float Tensor / Numpy ndarray): MLE of P(Y=1)

    '''
    return (torch.sum(y == 1) / y.shape[0]) # Y=1 / total
    

def bayes_classify(D, p, X):
    '''
    Arguments:
        D (2 x N Float Tensor / Numpy ndarray): returned value of `bayes_MAP`
        p (float or scalar Float Tensor / Numpy ndarray): returned value of `bayes_MLE`
        X (S x N LongTensor / Numpy ndarray): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor / Numpy ndarray): label of each object for classification, y[i] = 0/1
         
    '''
    S = X.shape[0] #the number of objects
    N = D.shape[1] #the number of features
    result = torch.zeros(S)

    # loop to classify each data
    for i in range(0, S):
         prob_0 = 1 #initial (Y=0) = 1
         prob_1 = 1 # initial (Y=1) = 1
       
         for j in range(0,N):
             if X[i,j] == 0:
                 prob_0 *= (1-D[0,j]) #update the likelihood (Y=0)
                 prob_1 *= (1-D[1,j]) #update the likelihood (Y=1)
             elif X[i,j] == 1:
                 prob_0 *= D[0,j] #updates the likelihood of the ith object belonging to class 0
                 prob_1 *= D[1,j] ##updates the likelihood of the ith object belonging to class 1
                 
    
         if torch.log(prob_1 * p) > torch.log(prob_0 * (1-p)):
                result[i] = 1
         else:
                result[i] = 0
        
    return result           

# Problem Gaussian Naive Bayes


def gaussian_MAP(X, y):
    '''
    Arguments:
        X (S x N FloatTensor / Numpy ndarray): features of each object
        y (S LongTensor / Numpy ndarray): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor / Numpy ndarray): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor / Numpy ndarray): MAP estimation of mu in N(mu, sigma2)

    '''
    
    S = X.shape[0] # number of objects
    N = X.shape[1] # number of features
    result = torch.zeros((2, N))
    X_0 = X[y == 0] #number of instance y = 0
    X_1 = X[y == 1] #number of instance y = 1

    #get the mean and variance of the two class
    mu_0 = torch.mean(X_0, dim = 0)
    # expecting a value of 1.0, add unbiased = False
    sigma2_0 = torch.var(X_0, dim = 0,unbiased=False)
    
    mu_1 = torch.mean(X_1, dim = 0)
    sigma2_1 = torch.var(X_1, dim = 0,unbiased=False)
    
    # combine with the means and variance
    mu= torch.stack([mu_0, mu_1])
    sigma2 = torch.stack([sigma2_0,sigma2_1])
    
    return mu, sigma2
    

def gaussian_MLE(y):
    '''
    Arguments:
        y (S LongTensor / Numpy ndarray): label of each object

    Returns:
        p (float or scalar Float Tensor / Numpy ndarray): MLE of P(Y=1)

    '''
    return (torch.sum(y == 1) / y.shape[0]) # Y=1 / total
    

def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor / Numpy ndarray): returned value #1 of `gaussian_MAP` (estimation of mean)
        sigma2 (2 x N Float Tensor / Numpy ndarray): returned value #2 of `gaussian_MAP` (square of sigma)
        p (float or scalar Float Tensor / Numpy ndarray): returned value of `bayes_MLE`
        X (S x N LongTensor / Numpy ndarray): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor / Numpy ndarray): label of each object for classification, y[i] = 0/1
    
    '''
    S = X.shape[0] #the number of objects
    N = X.shape[1] #the number of features
    result = torch.zeros(S)
    import torch

def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor): Estimation of mean for each feature and each class.
        sigma2 (2 x N Float Tensor): Estimation of variance for each feature and each class.
        p (float): Prior probability for class 1.
        X (S x N Float Tensor): Feature matrix where each row represents an object and each column a feature.
        
    Returns:
        y (S LongTensor): Predicted class labels for each object in X.
    '''
    N = X.shape[1]
    S = X.shape[0]
    result = torch.zeros(S)
    # For each data point, estimate its classification
    for i in range(S):
        #Probabilities for each class belonging to 0 and 1
        probs = torch.zeros(2)
        # Loop the two classes (0 and 1)
        for y in range(2):
        # Initialized to zeros
            sum = 0
            # Compute Prior Log Probabilities:
            # Prior probability log(p) for class 1 and log(1-p) for class 0
            if y == 1:
                sum += torch.log(p)
            else:
                sum += torch.log(1 - p)
            # loop over features
            for j in range(N):
            #the probability density function (PDF)
                sum += (torch.log(1 / (torch.sqrt(2 * torch.pi * sigma2[y, j]))) - 
                        0.5 * ((X[i, j] - mu[y, j])**2 / sigma2[y, j]))

            probs[y] = sum

        # On maximum log probability
        result[i] = torch.argmax(probs)

    return result








   