#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cPickle as pickle
def prepareData(input_path):
    fp = open(input_path,'r')
    users = {}; items = {}
    lu = 0 ; li = 0;
    lines = fp.readlines()
    for line in lines:
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
    for line in lines:
        line = line.strip().split()
        u = line[0]
        i = line[1]
        r = line[2]
        u_i_mat[users[u]][items[i]] = r
        print u,i,u_i_mat[users[u]][items[i]]
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
    buVec = np.zeros(m)
    biVec = np.zeros(n)
    rating_num = 0
    sum = 0
    for i in range(m):
        for j in range(n):
            if rMat[i][j] != 0:
                rating_num += 1
                sum += rMat[i][j]
    mu = sum/rating_num
    print mu
    #high parameters
    epoch = 5; rate = 0.005; lam = 0.02

    print m,n
    for k in range(epoch):
        loss = 0
        for i in range(m):
            for j in range(n):
                if rMat[i][j] != 0:
                    pu = uMat[i]
                    qi = iMat[j]
                    pu_qi = pu.dot(qi.T)
                    e_ui = rMat[i][j] - mu - buVec[i] - biVec[j] - pu_qi
                    loss += e_ui*e_ui
                    uMat[i] = uMat[i] + rate*(e_ui*iMat[j]-lam*iMat[j])
                    loss += lam*((uMat[i].sum())**2)
                    iMat[j] = iMat[j] + rate*(e_ui*uMat[i]-lam*uMat[i])
                    loss += lam*((iMat[j].sum())**2)
                    buVec[i] = buVec[i] + rate*(e_ui-lam*buVec[i])
                    loss += lam*(buVec[i]**2)
                    biVec[j] = biVec[j] + rate*(e_ui-lam*biVec[j])
                    loss += lam*(biVec[j]**2)
                    loss *= 0.5
        print "The %d epoch loss is %f"%(k,loss)

if __name__ == "__main__":
    #input = "/home/jack/workspace/graduation_project/data/ml-100k/u.data"
    #prepareData(input) 
    matrixFac()
