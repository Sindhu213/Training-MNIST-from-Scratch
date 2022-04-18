from src import tools
import numpy as np

def init_params():
    w1 = np.random.randn(10,784)
    b1 = np.random.randn(10,1)
    w2 = np.random.randn(10,10)
    b2 = np.random.randn(10,1)
    return w1,b1,w2,b2

def forward_pass(w1,b1,w2,b2,X,alpha=0.2):
    z1 = w1.dot(X)+b1
    A1 = tools.Leaky_ReLU(z1,alpha)
    z2 = w2.dot(A1)+b2
    A2 = tools.softmax(z2)
    return z1, A1, z2, A2

def backward_pass(z1,a1,a2,w2,X,Y,alpha=0.2):
    one_hot_Y = tools.one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = (1/Y.shape[1])*dz2.dot(a1.T)
    db2 = (1/Y.shape[1])*np.sum(dz2,axis=1,keepdims=True)
    dz1 = w2.T.dot(dz2) * tools.Leaky_ReLU_derivative(z1,alpha)
    dw1 = (1/Y.shape[1])*dz1.dot(X.T)
    db1 = (1/Y.shape[1])*np.sum(dz1,axis=1,keepdims=True)
    return dw1, db1, dw2, db2

def update_params(w1,dw1,b1, db1,w2,dw2,b2,db2,eta):
    w1 = w1 - eta*dw1
    b1 = b1 - eta*db1
    w2 = w2 - eta*dw2
    b2 = b2 - eta*db2
    return w1,b1,w2,b2

def get_prediction(A2):
    return np.argmax(A2,axis= 0 )

def get_accuracy(predictions, y):
    print(f"Prediction, Actual = {predictions}, {y}")
    return np.sum(predictions == y)/y.shape[1]

def gradient_descent(X,y,learning_rate,epochs):
    w1, b1, w2, b2 = init_params()
    for i in range(epochs):
        z1, a1, z2, a2 = forward_pass(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = backward_pass(z1, a1, a2, w2, X, y)
        w1, b1,w2,b2 = update_params(w1,dw1,b1, db1,w2,dw2,b2,db2,learning_rate)
        if i%10 == 0:
            print("Epoch: ",i)
            predictions = get_prediction(a2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2