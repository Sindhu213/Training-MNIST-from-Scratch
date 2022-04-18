import numpy as np
from src import tools

class Neural_Network:
    """Generalized Neural Network for classification based problems"""

    def __init__(self,sizes):
        """ Sizes is the list of numbers representing number of units in each layer """
        self.sizes = sizes
        self.layers = len(sizes)
        self.weights = [np.random.randn(fan_out, fan_in) for fan_in, fan_out in zip(self.sizes[:-1],self.sizes[1:])]
        self.biases = [np.random.randn(fan_out,1) for fan_out in self.sizes[1:]]


    def feedforward(self,X,alpha=0.3):
        """Returns the output of forward propagation"""
        for w,b in zip(self.weights, self.biases):
            X = tools.Leaky_ReLU(w.dot(X) + b,alpha)
        return X


    def forward_pass(self,X,alpha=0.3):
        """Returns the list containing values of z_s and a_s using forward propagation"""
        a_s, z_s = [X], []
        activation = X
        for w,b in zip(self.weights, self.biases):
            z = w.dot(activation) + b
            z_s.append(z)
            activation = tools.Leaky_ReLU(z,alpha)
            a_s.append(activation)
        return z_s,a_s


    def backward_pass(self,X,y):
        """Returns the partial derivatives of cost function wrt to parameters using backpropagation"""
        z_s, a_s = self.forward_pass(X)
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        delta = tools.softmax_derivative(z_s[-1],y)
        delta_w[-1] = delta.dot(a_s[-2].transpose())
        delta_b[-1] = np.sum(delta,axis=1, keepdims=True)

        for i in range(2,self.layers):
            delta = np.dot(self.weights[-i+1].transpose(),delta) * tools.Leaky_ReLU_derivative(z_s[-i])
            delta_w[-i] = delta.dot(a_s[-i-1].transpose())
            delta_b[-i] = np.sum(delta,axis=1, keepdims=True)
        return delta_w,delta_b


    def optimize(self,X,y,eta=0.08,epochs = 500):
        """optimize cross entropy loss function using gradient descent"""
        for i in range(epochs):
            dw, db = self.backward_pass(X,y)
            self.weights = [w - (eta/y.shape[1])* _dw for w, _dw in zip(self.weights,dw)]
            self.biases = [b - (eta/y.shape[1])* _db for b, _db in zip(self.biases,db)]
            if i % 10 == 0:
                print("Epoch: ", i)
                predictions = self.get_prediction(self.feedforward(X))
                print(self.get_accuracy(predictions, y))
        return self.weights, self.biases


    @staticmethod
    def get_prediction(A2):
        return np.argmax(A2, axis=0)


    @staticmethod
    def get_accuracy(predictions, y):
        print(f"Prediction, Actual = {predictions}, {np.argmax(y,axis=0)}")
        return np.sum(predictions == np.argmax(y,axis=0)) / y.shape[1]


    def evaluate(self, X_test, y_test):
        return self.get_accuracy(self.get_prediction(self.feedforward(X_test)),y_test)



