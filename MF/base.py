#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import cPickle as pickle
def prepareData(input_path):
    fp = open(input_path,'r')
    users = {}; items = {}
    lu = 0 ; li = 0;
    for line in fp.readlines():
        line = line.strip().split()
        u = line[0]
        i = line[1] 
        if u not in users:
            users[u] = lu
            lu += 1
        if i not in items:
            items[i] = li
            li += 1
    u_i_mat = np.zeros((lu,li))
    for line in fp.readlines():
        line = line.strip().split()
        u = line[0]
        i = line[1]
        r = line[2]
        u_i_mat[users[u]][items[i]] = r
    tmpdatapath = "/home/jack/workspace/graduation_project/tmpdata/u_i_matrix.csv"
    tmpusers = "/home/jack/workspace/graduation_project/tmpdata/users.csv"
    tmpitems = "/home/jack/workspace/graduation_project/tmpdata/items.csv"
    savefp = open(tmpdatapath,"w")
    save_u = open(tmpusers,"w")
    save_i = open(tmpitems,"w")
    pickle.dump(u_i_mat,savefp) 
    pickle.dump(users,save_u)
    pickle.dump(items,save_i)
    fp.close();savefp.close();save_u.close();save_i.close()
    print "prepareData finished"

#def saveData(data):

def matrixFac():
    fp_u = open("/home/jack/workspace/graduation_project/tmpdata/users.csv","r")
    fp_i = open("/home/jack/workspace/graduation_project/tmpdata/items.csv","r")
    fp_mat = open("/home/jack/workspace/graduation_project/tmpdata/u_i_matrix.csv","r")    
    users = pickle.load(fp_u)
    items = pickle.load(fp_i)
    rMat = pickle.load(fp_mat)
    m,n = rMat.shape    
    #k is the latent dimension
    k = 20
    uMat = np.random.randn(m,k)
    iMat = np.random.randn(n,k)
    buVec = np.mat(zeros(m))
    biVec = np.mat(zeros(n))
    rating_num = 0
    for i in rMat:
        if i != 0:
            rating_num += 1
    mu = rMat.sum()/rating_num
    #high parameters
    epoch = 100; rate = 0.005; lam = 0.02

    rmse_3 = np.asarray([0,0,0])
    for k in epoch:
        errsum = 0
        for i in range(m):
            for j in range(n):
                if rMat[i][j] != 0:
                    e_ui = rMat[i][j] - mu - buVec[i] - biVec[j] - uMat[i].dot(uMat[:][j])
                    uMat[i] = uMat[i] - rate*(e_ui+lam*iMat[j])
                    iMat[j] = iMat[j] - rate*(e_ui+lam*uMat[i])
                    buVec[i] = buVec[i] - rate*(e_ui+lam*buVec[i])
                    biVec[j] = biVec[j] - rate*(e_ui+lam*biVec[j])
                    errsum += e_ui**2
        rmse = (errsum/rating_num)**0.5
        print "The %d epoch rmse is %f"%(k,rmse)
        break
        if rmse_3.sum()/3 - rmse < 0.0001:
            break
        rmse_3[k%3] = rmse 

if __name__ == "__main__":
    input = "/home/jack/workspace/graduation_project/data/ml-100k/u.data"
    prepareData(input) 
