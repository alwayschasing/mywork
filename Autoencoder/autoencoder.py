#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import math

class Autoencoder(object):
    #对该类的训练以及transform函数调用必须在一个session之内
    def __init__(self,dimensions):
        #例如[6,3,2],列表形式，三层，输入层大小为6
        self.dim = dimensions 
        
        #建立网络结构
        """
        x: Tensor 
            Input placeholder to the network
        h: Tensor 
            latent representation
        y: Tensor 
            output reconstruction of the input
        """
        #input to the network
        self.x = tf.placeholder(tf.float32,[None,dimensions[0]],name='input_x')
        current_input = self.x

        #build the encoder
        encoder = []
        for layer_i,n_output in enumerate(dimensions[1:]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(tf.random_uniform([n_input,n_output],
                                             -1.0/math.sqrt(n_input),
                                             1.0/math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            output = tf.nn.tanh(tf.matmul(current_input,W)+b)
            current_input = output

        #latent representation
        self.h = current_input

        #如果解码过程与编码工程使用相同的权值参数
        encoder.reverse()

        ##build the decoder using the same weights
        #for layer_i,n_output in enumerate(dimensions[:-1][::-1]):
            #W = tf.transpose(encoder[layer_i])
            #b = tf.Variable(tf.zeros([n_output]))
            #output = tf.nn.tanh(tf.matmul(current_input,W)+b)
            #current_input = output
        decoder = []
        for layer_i,n_output in enumerate(dimensions[-2::-1]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(tf.random_uniform([n_input,n_output],
                                             -1.0/math.sqrt(n_input),
                                             1.0/math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            decoder.append(W)
            output = tf.nn.tanh(tf.matmul(current_input,W)+b)
            current_input = output

        #now have the reconstruction through the network
        Wo = tf.Variable(tf.random_uniform(
            [dimensions[0],dimensions[0]],
            -1.0/math.sqrt(dimensions[0]),
            1.0/math.sqrt(dimensions[0])
            ))
        self.logits = tf.matmul(current_input,Wo)+b

        self.y = tf.nn.sigmoid(self.logits)

        #cost function measures pixel-wise difference
        #self.cost = tf.reduce_sum(tf.square(self.y-self.x))
        self.cost = tf.contrib.losses.sigmoid_cross_entropy(
            multi_class_labels = self.x,
            logits = self.logits
            )
        
    #训练函数，参数有session,训练数据集
    #调用时需要给出optimizer
    def fit(self,sess,train_data,optimizer,n_epoch=100,batch_size=100):

        sess.run(tf.global_variables_initializer()) 
        optimizer = optimizer.minimize(self.cost)
        #训练集的总长度
        size = len(train_data)
        batch_begin = 0
        #训练过程 
        for i in range(size//batch_size):
            batch_data = train_data[batch_begin:batch_begin+batch_size*(i+1)]
            sess.run(optimizer,feed_dict={self.x:batch_data})
        #训练最后一个batch,存在余下的不够一个batch的情况
        last = train_data[-batch_size:]
        sess.run(optimizer,feed_dict={self.x:last})
        last_cost = sess.run(self.cost,feed_dict={self.x:last})
        print "the %d epoch cost is %f"%(0,last_cost)

        for epoch in range(1,n_epoch):
            #每轮训练按batch进行优化,加快速度
            for i in range(size//batch_size):
                batch_data = train_data[batch_begin:batch_begin+batch_size*(i+1)]
                sess.run(optimizer,feed_dict={self.x:batch_data})
            #训练最后一个batch,存在余下的不够一个batch的情况
            last = train_data[-batch_size:]
            sess.run(optimizer,feed_dict={self.x:last})
            cost = sess.run(self.cost,feed_dict={self.x:last})
            print "the %d epoch cost is %f"%(epoch,cost)
            #if math.fabs(last_cost-cost)<10:
                #break

    #对要编码的数据集进行编码
    def transform(self,sess,x_input):
        return sess.run(self.h,feed_dict={self.x:x_input})


