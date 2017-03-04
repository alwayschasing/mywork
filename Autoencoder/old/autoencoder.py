#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

def randominit(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),
                            minval = low,maxval = high,
                            dtype = tf.float32)
class Autoencoder(object):
    
    def __init__(self,n_input,n_hidden,activation=tf.nn.softplus,optimizer=tf.train.GradientDescentOptimizer(learning_rate = 0.01)):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation = activation

        network_weights = self.initialize_weights()
        self.weights = network_weights
        
        #model 
        self.x = tf.placeholder(tf.float32, [None,self.n_input])
        self.hidden = self.activation(tf.add(tf.matmul(self.x,self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.sigmoid(tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2']))

        #cost 
        self.cost = -tf.reduce_sum(tf.mul(self.x,tf.log(self.reconstruction)))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(randominit(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]),dtype = tf.float32)
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    def fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X})
        return cost
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X})



