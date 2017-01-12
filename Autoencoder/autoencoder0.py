#!/usr/bin/env python
# coding=utf-8

import numpy as np

def sigmoid(z):
    """the sigmoid function"""
    return 1.0/(1.0+np.exp(-z))
"""
def softmax(x,wlist):
    sum = 0.0
    res = []
    for w in wlist:
        sum += np.exp(dot(x,w))
    for w in wlist:
        res.append(np.exp(dot(x,w)/sum))
    return res
"""

def softmax(input,w,b):
    """
    the softmax layer is constituted of many softmax output node
    w has 3 dimisions,for example [n][hidden_size][5]
    in has the shape like [hidden_size][5]
    output has the shape like [n][5]
    """
    output = np.zeros((w.shape[0],w.shape[2]))
    n = output.shape[0]
    for i in xrange(n):
        tmp = np.dot(input.T,w[i])+b[i]
        sum = np.sum(tmp)
        output[i] = [item/sum for item in tmp]
    return output
         

class CrossEntropyCost(object):
    @staticmethod
    def get(y,y_hat):
        return np.sum(np.nan_to_sum(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)))

class LogCost(object):
    @staticmethod
    def get(y,y_hat):
        return -np.log(y_hat)

class AutoEncoder(object):

    def __init__(self,input_size,hidden_size,input):
        """input is [n][5]""" 
        self.inpt = input
        self.inpt_size = input_size
        self.h_size = hidden_size
    
    #generate the corrupted input for denoised Autoencoder
    def corruptInput(self):
        return 0

    def build(self,activation=sigmoid):
        self.w1 = np.random.randn(self.inpt[0],5,self.h_size)
        self.b1 = np.random.randn(self.h_size,1) 
        self.w2 = np.random.randn(self.inpt[0],self.h_size,5)
        self.b2 = np.random.randn(self.inpt[0],1)
        
    """
    def feedforward(self,a):
        #return the output of the layer ahead of last output layer
        for w,b in zip(self.weights,self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a
    """
    def backpropagation(self):
        """nabla_w,nabla_b are gradients for cost"""
        nabla_w1 = np.zeros(self.w1.shape)
        nabla_b1 = np.zeros(self.b1.shape)
        nabla_w2 = np.zeros(self.w2.shape)
        nabla_b2 = np.zeros(self.b2.shape)
        #the feedforward progress
        z1 = np.zeros(self.h_size)
        for i in xrange(self.inpt.shape[0]):
            z1 += np.dot(self.inpt[i],self.w1[i])
        a1 = sigmoid(z1) #here a1 is a ndarray[hidden_size][1]

        y_hat = softmax(a1,self.w2,self.b2)
        y = self.inpt
        #the backpropagation
        for i in xrange(self.w2.shape[0]):
            nabla_w2[i] = np.dot(a1,(y_hat-y)[i])
            nabla_b2[i] = np.dot(y_hat-y)[i]
        for i in xrange(self.w2.shape[0]):
            nabla_w1[i] += np.dot(self.w2[i].T, (y_hat-y))*a1(1-a1)*self.inpt[i]
            nabla_b1[i] += np.dot(self.w2.T,(y_hat-y))*a1(1-a1)
        return nabla_w1,nabla_b1,nabla_w2,nabla_b2

    def train(self):
        return 0
        
        

    def cost(self,y,y_hat):
        """
        output layer constitutes many 5 size of unit sotmax
        y has two dimisions ,such as y[n][5],n denote n moives,5 denote there are 
        5 kind of ratings
        """
        C = np.zeros(y.shape[0])
        ktypes = y.shape[1] #here is 5
        for i in xrange(y.shape[0]):
            for j in xrange(ktypes):
                if y[i][j] == 1:
                    C[i] = -np.log(y_hat[j])
                    break
        return np.sum(C)

if __name__ == "__main__":
    print "Autoencoder"
        
