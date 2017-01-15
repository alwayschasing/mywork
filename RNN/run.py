#!/usr/bin/env python
# coding=utf-8

import cPickle as pickle 
import numpy as np
import csv
from rnn import LstmModel

rootdir = "/home/jack/workspace/graduation_project/"
neighbor_k = 10

#generate sequence data from autoencoder data
def generateDataFromAutoencoder():
    fpiv = open(rootdir+"data/ml-1m/autoencoder/item_hidden_vector","rb")
    item_vectors = pickle.load(fpiv)
    fpid = open(rootdir+"data/ml-1m/autoencoder/item_index","rb")
    item_index = pickle.load(fpid)
    fprd = open(rootdir+"data/ml-1m/rnndata.csv","r")
    reader = csv.reader(fprd)
    sequences = list()
    for line in reader:
        seq = list()
        for i in line:
            seq.append(item_vectors[item_index[i]])
        sequences.append(seq)
    fpiv.close();fpid.close();fprd.close()
    return sequences

def generateDataFromMF():
    pass

def generateDataFromEmbedding():
    pass

def generateTestData():
    path = rootdir+"data/ml-1m/"
    input = list()
    targetlist = list()
    with open(path+"userbased.train.csv","r") as fp:
        inputlines = csv.reader(fp)
        for line in inputlines:
            line = line[-9:] 
            input.append(line)
    with open(path+"userbased.test.csv","r") as fptest:
        targetlines = csv.reader(fptest)
        for line in targetlines:
            targetlist.append(line)
    input = np.asarray(input)
    targetlist = np.asarray(targetlist)
    return input,targetlist

      
def trainRNN(traindata,n_step,hidden_size,lr):
    #shape shoud be (batch_size,n_step,hidden_size)
    batch_size = len(traindata)

    x = traindata[:,:-1,:]
    y = traindata[:,1:,:]
    print batch_size
    print y.shape
    print y.shape

    rnn_model = LstmModel(batch_size,n_step,hidden_size,lr)
    training_epoch = 100
    rnn_model.train(x,y,training_epoch)
    return rnn_model

# def pred(model,data):
#     res = model.pred(data)
#     return res

#test on the testdata by recall
def test(model,input,targets,dataset):
    pred_res = model.pred(input)
    #the pred_items is lists of actual item ids for each user
    pred_items_indices=knn(dataset,pred_res)
    pred_items = list()
    with open(rootdir+"data/autoencoder/index2item","rb") as fp_id2item:
        index2item = pickle.load(fp_id2item)
        for each_u in pred_items_indices:
            ubasedpred = list()
            for i in each_u:
                ubasedpred.append(index2item[i])
            pred_items.append(ubasedpred)
    recalls = np.zeros((input.shape[0],))
    n_valid = 0
    denominator = float(neighbor_k)
    for u,ilists in enumerate(pred_items):
        if len(targets[u]) < neighbor_k: continue
        n_valid += 1
        hit = 0
        for v in ilists:
            if v in targets[u]: hit += 1
        recalls[u] = hit/denominator
        
    return recalls.sum()/n_valid

def knn(dataset,vec_list):
    reslist = list() 
    totalsize = dataset.shape[0]
    dists = np.zeros((totalsize,))
    k = neighbor_k
    for vec in vec_list:
        for index,item in enumerate(dataset):
            dists[index] = ((item-vec)**2).sum()
        
        sortedDistIndices = dists.argsort()
        reslist.append(sortedDistIndices[:k])
    return reslist

def main():
    traindata = generateDataFromAutoencoder()
    traindata = np.asarray(traindata)

    model = trainRNN(traindata,n_step=9,hidden_size=200,lr=0.05)
    del traindata

    fpiv = open(rootdir + "data/ml-1m/autoencoder/item_hidden_vector", "rb")
    item_vectors = pickle.load(fpiv)
    testdata, targets = generateTestData()
    recall = test(model,testdata,targets,item_vectors)
    print "recall is %f"%recall

if __name__ == "__main__":
    main()





