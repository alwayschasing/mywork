#!/usr/bin/env python
# coding=utf-8
import csv
import numpy as np
import sys
import time
import random

"""
用来得到用于矩阵分解的rating矩阵
"""
def getRatingMatrix():
    infp = open("../userbased.trainMF.csv","r")
    outfp = open("./trainForMFlib","w")

    readerTrain = csv.reader(infp)
    for line in readerTrain:
        n_item = len(line)
        #每行第一个是用户编号,然后是一对对item与rating
        for i in range(n_item//2):
            #the index of item,next is rating
            k = i*2+1
            outfp.write(line[0]+" "+line[k]+" "+line[k+1]+"\n")
    infp.close()
    outfp.close()


    intestfp = open("../userbased.testMF.csv","r")
    otestfp = open("./testForMFlib","w")

    readerTest = csv.reader(intestfp)
    for line in readerTest:
        n_item = len(line)
        for i in range(n_item//2):
            k = i*2+1
            otestfp.write(line[0]+" "+line[k]+" "+line[k+1]+"\n")

    intestfp.close()
    otestfp.close()

"""
用来得到用于矩阵分解的item共现矩阵的数据
"""
def getCoCurrenceData():
    #input file point 
    infp = open("../userbased.train.csv","r")
    lines = infp.readlines()

    #找出索引值最大的item，以便建立矩阵
    maxItem_index = 0
    for line in lines:
        line = line.strip().split(',')
        for i in line[1:]:
            if int(i) > maxItem_index:
                maxItem_index = int(i)

    print "max item index is %d"%maxItem_index

    coCurMat = np.zeros((maxItem_index+1,maxItem_index+1),dtype=np.int32)

    for line in lines:
        line = line.strip().split(',')
        line = np.asarray(line,dtype=np.int32)
        for i in line[1:]:
            for j in line[1:]:
                if i != j:
                    coCurMat[i][j] += 1
    print "has build the matrix"
    outfp = open("./coCurMatTrain","w")
    len = coCurMat.shape[0]
    for i in range(len):
        for j in range(len):
            if coCurMat[i][j] != 0:
                print>>outfp,"%d %d %d"%(i,j,coCurMat[i][j])

    outfp.close()

#产生训练分解item共现矩阵模型的训练集与测试集
def getCoMFtrain_testData():
    fp = open("./coCurMatTrain","r")
    lines = fp.readlines()
    otrain = open("./coCurTr","w")
    otest = open("./coCurTe","w")
    #矩阵的维度
    n = len(lines)
    for i  in range(n//10):
        k = random.randint(0,9)
        for j in range(10):
            if j == k:
                otest.write(lines[10*i+j])
            else:
                otrain.write(lines[10*i+j])
    for i in range(n%10):
        otrain.write(lines[10*i+j])
    fp.close()
    otrain.close()
    otest.close()
    print "finish coCur train and test data generation"


    
    
if __name__ == "__main__":
    start = time.time()
    #getCoCurrenceData()
    getCoMFtrain_testData()
    #if sys.argv[1] == '1':
        #print "get the rating matrix data"
        #getRatingMatrix()
    #elif sys.argv[1] == '2':
        #print "get the cocurrence data"
        #getCoCurrenceData()
    #elif sys.argv[1] == '3':
        #print "get coCur train and test data generation"
        #getCoMFtrain_testData()
    #else:
        #print "input the correct parameter"
    end = time.time()
    print "has ran %d miniutes"%((end-start)/60)
