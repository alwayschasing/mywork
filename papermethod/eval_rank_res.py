#!/usr/bin/env python
# coding=utf-8
import csv
import numpy as np

def evaluate():
    """
    默认每个的候选推荐列表为30
    """
    fp = open("/home/lrh/graduation_project/data/ml-1m/paperTestdata.csv","r")
    reader = csv.reader(fp)
    te_data = list(reader)
    te_data = np.asarray(te_data,dtype=np.int32)
    #首项为用户编号
    indexs = [0,19,20,21,22,23,24,25,26,27,28]
    te_target = te_data[:,19:]

    fpitem = open("/home/lrh/graduation_project/data/ml-1m/rnn_rec_res.csv","r")
    fpr = open("/home/lrh/graduation_project/data/ml-1m/rating_res","r")

    recommend_set = []
    back_rec_size = 20
    k = 0
    tmp = []
    for line in fpitem.readlines():
        line = line.strip().split()
        tmp.append(line[1]) #只要item编号
        k += 1
        if k == back_rec_size:
            recommend_set.append(tmp)
            tmp = []
            k = 0

    ratings = []
    tmp_r = []

    k = 0
    for line in fpr.readlines():
        line = line.strip().split()
        tmp_r.append(line[0])
        k += 1
        if k == back_rec_size:
            ratings.append(tmp_r)
            tmp_r = []
            k = 0
        
    """
    recommend_set与ratings的形式都为:
    每个用户一个大小为30的list,所有用户list构成整体
    """    

    recommends = np.asarray(recommend_set,dtype=np.int32)
    ratings = np.asarray(ratings,dtype=np.float32)

    indexs = np.argsort(ratings,axis=-1)

    n = te_target.shape[0]

    fin_recommend = []
    for i in xrange(n):
        fin_recommend.append(recommends[i][indexs[i][-10:]])

    fin_recommend = np.asarray(fin_recommend,dtype=np.int32) 

        
    hit = 0
    hituser = 0
    recall = 0.0
    for i in xrange(n):
        hit = 0
        for item in fin_recommend[i]:
            if item in te_target[i]:
                hit += 1
        if hit != 0:
            hituser += 1
        recall += float(hit)/10

    print "hituser:%d"%hituser
    rec = recall/te_target.shape[0]
    print "recall:%f"%rec

if __name__=="__main__":
    evaluate()
