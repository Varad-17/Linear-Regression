import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch

class neural_network :
    
    def __init__(self, activation_fn, derivative_fn, hidden,n_input = 2, layers_hidden = 1, n_output = 1):
        self.inp = n_input
        self.layers = layers_hidden
        self.out = n_output
        self.weights  = np.empty([self.layers+1], dtype = object)
        self.hidden = hidden
        self.activation = activation_fn
        self.derivative = derivative_fn
        
    def input_neurons(self):
        assert len(self.hidden) == self.layers, 'Size mismatch between number of hidden layers and the array containing neurons in each hidden layer'
        n_input = self.inp
        for i in range(self.layers):
            n_out = self.hidden[i]
            self.weights[i] = np.random.rand(n_out, n_input+1)
            n_input = n_out
        self.weights[-1] = np.random.rand(self.out, n_input+1)
                
    def train(self,data, truth_val, cost_fn, lr = 0.1, epochs = 1000):
        truth_val = truth_val.T
        for epoch in range(epochs):
            values = self.predict(data, train = True)
            gradient = np.empty([self.out,len(data)])
            gradient = values[-1] - truth_val
            cost = cost_fn(truth_val, values[-1][0])
            print(f'After {epoch+1}th epoch, the Value of the cost function is {cost} and the predicted output is {values[-1][0]}')
            #print(self.weights)
            for it in range(len(values)-1,-1,-1): 
                #print(gradient)
                for it2 in range(len(gradient)):
                    if (it - 1) < 0:
                        value = data.T                   
                    else :
                        value = values[it-1]
                    #print(gradient[it2],np.concatenate([value, np.ones((1,len(data)))]))
                    update = np.mean(gradient[it2] * np.concatenate([value, np.ones((1,len(data)))]), axis = 1)
                    #print(self.weights[it][it2], update)
                    self.weights[it][it2] = self.weights[it][it2] - lr*update
                gradient = self.derivative(values[it])*gradient
                gradient = np.matmul(self.weights[it][:,:-1].T,gradient)            
                     
    
    def predict(self, x, train = False):
        shape = len(x)
        x = x.T
        for r in range(len(self.weights)):
            x = np.concatenate([x,np.ones((1,shape))], axis = 0)
            prediction = self.activation(np.matmul(self.weights[r],x))
            x = prediction
            if r == 0:
                output = list(np.expand_dims(prediction, axis = 0))
            else :
                output.append(prediction)
        if train:
            return np.array(output, dtype = object)
        
        return x
     
