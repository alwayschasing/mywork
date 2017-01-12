#!/usr/bin/env python
# coding=utf-8

import numpy as np
import sys
import cPickle as pickle
from autoencoder import Autoencoder
import tensorflow as tf

fp = open("../data/ml-1m/itembased.train.csv","r")
lines = fp.readlines()
itemsize = len(lines)
#the index of user in array
userindex = pickle.load(open("../data/ml-1m/userindex","rb"))
n_user = 6040
training_epoch = 500
display_step = 10
precost = sys.maxint
encodernetwork = Autoencoder(n_input = n_user,
                            n_hidden = 200,
                            activation = tf.nn.softplus,
                            optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005))

X = np.zeros((itemsize,n_user),dtype = int)

#build the map item to index
item_index_X = dict()
#build the map index to item
index2item = dict()


for i in range(itemsize):
    line = lines[i].strip().split(',')
    item_index_X[line[0]] = i
    index2item[i] = line[0]
    l = len(line)
    for j in range(1,l):
        X[i][userindex[line[j]]] = 1

for epoch in range(training_epoch):
    cost = encodernetwork.fit(X)
    #avg_cost += cost/itemsize
    """
    if precost < avg_cost:
        print "training ends earlier"
        break
    else:
        precost = avg_cost
    """
    if epoch%display_step == 0:
        print "epoch %d cost=%f"%(epoch,cost)

item_hidden = encodernetwork.transform(X)
sp1 = open("../data/ml-1m/autoencoder/item_hidden_vector","wb")
sp2 = open("../data/ml-1m/autoencoder/item_index","wb")
sp3 = open("../data/ml-1m/autoencoder/index2item","wb")
pickle.dump(item_hidden,sp1)
pickle.dump(item_index_X,sp2)
pickle.dump(index2item,sp3)
sp1.close()
sp2.close()
sp3.close()


