import numpy as np
def softmax(X):
    e_x = np.exp(X - np.max(X))
    return e_x / np.sum(e_x,axis=0)

def softmax_derivative(X, y):
    return softmax(X) - y

def cross_entropy(y, y_hat):
    y = y.reshape(y.shape[0],1)
    y_hat = y_hat.reshape(y.shape[0],1)
    return -1*np.sum(np.dot(y.T, np.log(y_hat+1e-5)))

def Leaky_ReLU(X,alpha=0.1):
    return np.maximum(alpha*X,X)

def Leaky_ReLU_derivative(X,alpha=0.1):
    derivative = np.ones_like(X,dtype=float)
    derivative[X<0.0] = alpha
    return derivative

def one_hot(Y):
    m = Y.shape[0] if Y.shape[0]>1 else Y.shape[1]
    one_hot_vector = np.zeros((10, m))
    Y = Y.reshape(m,)
    one_hot_vector[Y,np.arange(m)] = 1
    return one_hot_vector

def cross_entropy_derivative(y,y_hat):
    y_hat_ = np.reciprocal(y_hat)
    return np.dot(y, y_hat_.transpose())

