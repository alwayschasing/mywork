#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cPickle as pickle
import sys

def saveData(data,path):
    with open(path,"w") as fp:
        pickle.dump(data,fp)

def prepareData(input_path):
    fp = open(input_path,'r')
    users = {}; items = {}
    # row2user = {}; col2item = {}
    lu = 0 ; li = 0;
    lines = fp.readlines()
    for line in lines:
        line = line.strip().split(",")
        u = line[0]
        if u not in users:
            users[u] = lu
            lu += 1
        itemnum = len(line)/2
        for k in range(0,itemnum):
            i = line[2*k+1] 
            if i not in items:
                items[i] = li
                li += 1
    
    u_i_mat = np.zeros((lu,li),dtype=int)
    for line in lines:
        line = line.strip().split(",")
        u = line[0]
        itemnum = len(line)/2
        for k in range(0,itemnum):
            i = line[2*k+1]
            r = line[2*k+2]
            u_i_mat[users[u]][items[i]] = r
            print u,i,u_i_mat[users[u]][items[i]]
    datapath = "/home/jack/workspace/graduation_project/tmpdata/"
    np.savetxt(datapath+"u_i_matrix.txt",u_i_mat)
    saveData(users,datapath+"user2row")
    saveData(items,datapath+"item2col")
    print "prepareData finished"


def matrixFac(latentdim):
    fp_mat = open("/home/jack/workspace/graduation_project/tmpdata/u_i_matrix.txt","r")    
    # rMat = pickle.load(fp_mat)
    rMat = np.loadtxt(fp_mat)
    m,n = rMat.shape    
    #k is the latent dimension
    k = int(latentdim)
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
    epoch = 20000; rate = 0.005; lam = 0.02

    print m,n
    preloss = sys.maxint
    for k in range(epoch):
        for i in range(m):
            for j in range(n):
                if rMat[i][j] != 0:
                    pu = uMat[i]
                    qi = iMat[j]
                    pu_qi = pu.dot(qi.T)
                    e_ui = rMat[i][j] - mu - buVec[i] - biVec[j] - pu_qi
                    uMat[i] = uMat[i] + rate*(e_ui*iMat[j]-lam*uMat[i])
                    iMat[j] = iMat[j] + rate*(e_ui*uMat[i]-lam*iMat[j])
                    buVec[i] = buVec[i] + rate*(e_ui-lam*buVec[i])
                    biVec[j] = biVec[j] + rate*(e_ui-lam*biVec[j])
        loss = 0
        for i in range(m):
            for j in range(n):
                if rMat[i][j] != 0:
                    err = rMat[i][j] - mu - buVec[i] - biVec[j] - uMat[i].dot(iMat[j])
                    loss += err**2 + lam*(buVec[i]**2 + biVec[j]**2 + (iMat[j]**2).sum() + (uMat[i]**2).sum())
        print "The %d epoch loss is %f"%(k,loss)
        if loss < preloss:
            preloss = loss
        else:
            print "convergeced,loss is %f\n"%loss
            break
    return iMat  
    
if __name__ == "__main__":
    #input = "/home/jack/workspace/graduation_project/data/ml-1m/userbased.trainMF.csv"
    #prepareData(input) 
    k = sys.argv[1] 
    imat = matrixFac(k)
    np.savetxt("./npimat"+k+".txt",imat)
