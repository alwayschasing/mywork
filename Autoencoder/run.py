#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cPickle as pickle
from autoencoder import Autoencoder
import tensorflow as tf

fp = open("../data/ml-1m/itembased.train.csv","r")
lines = fp.readlines()
itemsize = 3952
print itemsize


#在数据集中出现的用户数量
n_user = 6040

#训练数据集
X = np.zeros((itemsize+1,n_user+1),dtype = np.float32)
for line in lines:
    line = line.strip().split(',')
    for u in line[1:]:
        X[int(line[0])][int(u)] = 1

dimensions = [n_user+1,6040,600,10]
encoder = Autoencoder(dimensions)

#定义自动编码机的优化方法
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate)

#建立session训练自动编码机，并使用
train_epoch = 20 
sess = tf.Session()
batch_size = len(X)
encoder.fit(sess,X,optimizer,n_epoch=train_epoch,batch_size=batch_size)
encoder_res = encoder.transform(sess,X)
print type(encoder_res)
print encoder_res.shape
resfp = open("../data/ml-1m/autoencoder/encoderdata","wb")
pickle.dump(encoder_res,resfp)
resfp.close()

