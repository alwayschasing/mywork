#!/usr/bin/env python
#encoding=utf-8
import csv
import tensorflow as tf
import math
import numpy as np

class EmbeddingModel:
    #num_sampled:the number we use to do the negative sample
    def __init__(self,datapath,embedding_size,training_epoch,num_sampled,session):
        self.datapath = datapath
        self.embedding_size = embedding_size
        self.training_epoch = training_epoch
        self.num_sampled = num_sampled
        self.sess = session

    def generate_training_data(self):
        data = list()
        itemindex = dict()
        with open(self.datapath,"r") as fp:
            datalines = csv.reader(fp)
            for line in datalines: 
                data.append(line)

        num = 0
        for i in data:
            for j in i:
                if j in itemindex: continue
                itemindex[j] = num
                num += 1
        #count the item_library size
        self.item_library_size = num + 1
        
        batch = list()
        labels = list()
        for window in data:
            for i in window:
                for j in window:
                    if j == i: continue
                    else:
                        batch.append(itemindex[i])
                        labels.append(itemindex[j])
        return batch,labels

    def buildModel(self,batch_size): 

        self.train_inputs = tf.placeholder(tf.float32,shape=[batch_size])
        self.train_labels = tf.placeholder(tf.float32,shape=[batch_size,1])

        #look up embeddings for inputs
        self.embeddings = tf.random_uniform([self.item_library_size,self.n_dim],-1.0,1.0)
        embed = tf.nn.embedding_lookup(self.embeddings,self.train_inputs)

        nce_weights = tf.Variable(tf.truncated_normal([self.item_library_size,self.embedding_size],
                                                     stddev=1.0/math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.item_library_size]))
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # using the noise-contrastive estimation training loss
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                        biases=nce_biases,
                        labels=self.train_labels,
                        inputs=embed,
                        num_sampled=self.num_sampled,
                        num_classes=self.item_library_size
                        )
            )

        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        self.init = tf.global_variables_initializer()
 
    def train(self):
        batch,labels = self.generate_training_data()
        self.init.run()
        for i in range(self.training_epoch):
            _,loss=self.sess.run([self.optimizer,self.loss],feed_dict={self.train_inputs:batch,self.train_labels:labels})
 
    def getEmbeddings(self):
        return self.embeddings 
        

