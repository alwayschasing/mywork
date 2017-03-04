#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cPickle as pickle
from autoencoder import Autoencoder
import tensorflow as tf

fp = open("../data/ml-1m/itembased.train.csv","r")
lines = fp.readlines()
itemsize = len(lines)

#build the map item to index
item_index = dict()
#build the map index to item
index2item = dict()

user_index = dict()
n_u = 0
for line in lines:
    line = line.strip().split(',')
    l = len(line)
    for i in range(1,l):
        if user_index.has_key(line[i]):
            continue
        else: user_index[line[i]] = n_u 
        n_u += 1

#在数据集中出现的用户数量
n_user = len(user_index)

#训练数据集
X = np.zeros((itemsize,n_user),dtype = int)
for i in range(itemsize):
    line = lines[i].strip().split(',')
    item_index[line[0]] = i
    index2item[i] = line[0]
    l = len(line)
    for j in range(1,l):
        X[i][user_index[line[j]]] = 1

dimensions = [6040,20]
encoder = Autoencoder(dimensions)

#定义自动编码机的优化方法
learning_rate = 0.005
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate)

#建立session训练自动编码机，并使用
train_epoch = 1000 
sess = tf.Session()
batch_size = len(X)
encoder.fit(sess,X,optimizer,n_epoch=train_epoch,batch_size=batch_size)
encoder_res = encoder.transform(sess,X)


sp1 = open("../data/ml-1m/autoencoder/item_hidden_vector","wb")
sp2 = open("../data/ml-1m/autoencoder/item_index","wb")
sp3 = open("../data/ml-1m/autoencoder/index2item","wb")
pickle.dump(encoder_res,sp1)
pickle.dump(item_index,sp2)
pickle.dump(index2item,sp3)
sp1.close()
sp2.close()
sp3.close()


