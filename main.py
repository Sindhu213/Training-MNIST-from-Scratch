import pandas as pd
from sklearn.model_selection import train_test_split

import src.tools
from src import ModelForMNIST,tools
from src import GeneralizedModel

#load and split data into training and test set
df = pd.read_csv('mnist-data/data.csv').to_numpy()
X, y = df[:,:-1], df[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=60000,random_state=42)
y_train_ = y_train.reshape(1,y_train.shape[0])
y_test_ = y_test.reshape(y_test.shape[0],1)

X_train_ = X_train.T/255.0

## training using ModelForMNIST
# W1, b1, W2, b2 = ModelForMNIST.gradient_descent(X_train_, y_train_, 0.80, 500)

## training using GeneralizedModel
Y_train_ = src.tools.one_hot(y_train_)
nn = GeneralizedModel.Neural_Network([784,10,10])
nn.optimize(X_train_,Y_train_,0.8)

## evaluating the model
print("Accuracy achieved on training data: ",nn.evaluate(X_test.T/255.0,src.tools.one_hot(y_test_)))


