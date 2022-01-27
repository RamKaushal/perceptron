import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OR = {"x1" : [0,0,1,1],
    "x2" : [0,1,0,1],
    "y":[0,1,1,1]
    }


class Perceptron:
    """
    desc: implementation of a peceptron in a ANN
    """
    def __init__(self,eta:float=None,epochs:int=None):
        self.eta = eta
        self.epochs = epochs 
        self.weights = np.random.rand(3)*1e-4 
    
    def z_outcome(self,inputs,weights):
        return np.dot(inputs,weights)

    def activation_function(self,z):
        return np.where(z>0,1,0)

    def fit(self,X,y):
        self.X = X
        self.y = y 
        X_with_bias = np.c_[self.X,-np.ones((len(self.X),1))]
        for epoch in range(self.epochs):
            z = self.z_outcome(X_with_bias,self.weights)
            print("weight in {} epoch is {}".format(epoch,z))
            yhat = self.activation_function(z)
            print("yhat in {} epoch is {}".format(epoch,yhat))

            error = self.y - yhat
            print("error in {} epoch is {}".format(epoch,error))

            self.weights +=  self.eta * np.dot(X_with_bias.T,error)
            print("updated weight in {} epoch is {}".format(epoch,self.weights))

        
       # return self.weights

    def predict(self,x):
        x_with_bias = np.c_(x,-np.ones((len(x),1)))
        pred = self.z_outcome(x_with_bias,self.weights)
        return self.activation_function(pred)

    def peparedata(self,name):
        self.df = pd.DataFrame(name)
        self.x = self.df.drop("y",axis = 1)
        self.y = self.df["y"]
        return self.x,self.y



if __name__ == "__main__":
    perceptron = Perceptron(0.01,10)
    x,y = perceptron.peparedata(OR)
    perceptron.fit(x,y)
    



