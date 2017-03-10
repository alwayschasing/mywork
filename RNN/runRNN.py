#!/usr/bin/env python
# coding=utf-8

import csv
import tensorflow as tf
import numpy as np
from rnn import LSTM

rootdir = "/home/lrh/graduation_project/"
neighbor_k = 10

def getMFData():
    fpin = open("/home/lrh/graduation_project/MF/MFModel","r")
    lines = fpin.readlines()
    num = len(lines)
    #
    max_n_item = 0
    for i in range(5,num):
        line = lines[i].split()
        flag = line[0][0]
        if flag == 'q':
            index = int(line[0][1:])
            if index > max_n_item:
                max_n_item = index
    latent_dim = 10
    #shape[0]加1是因为实际数据从1而不是从0开始
    item_latent_vec = np.zeros([max_n_item+1,latent_dim],dtype=np.float32)
    for i in range(5,num):
        line = lines[i].split()
        flag = line[0][0]
        if flag == 'q':
            index = int(line[0][1:])
            sign = line[1]
            if(sign=='T'):
                vec = line[2:]
                item_latent_vec[index] = vec
    print "has got MFdata"
    fpin.close()
    return item_latent_vec

def getTrainData():
    """
    返回的数据：涉及同一用户的序列数据组成一个列表，即为一个batch
    ,不同的batch大小不同，所有列表组成返回数据，例如对于用户1：
    [batch_size]
    return [n_user,batch_size(不同用户大小不一样),vec_size]
    """
    fpin = open("/home/lrh/graduation_project/data/ml-1m/rnndata.csv","r")
    lines = list(csv.reader(fpin))
    training_data = []
    n = len(lines)
    #表示用户编号
    u = lines[0][0]
    tmp = []
    for i in range(n):
        if lines[i][0] == u:
            tmp.append(lines[i])
        else:
            training_data.append(tmp)
            u = lines[i][0]
            tmp = [lines[i]] 
    training_data.append(tmp)
    fpin.close()
    return training_data

def getTestData():
    fp = open("/home/lrh/graduation_project/data/ml-1m/rnntestdata.csv","r")
    reader = csv.reader(fp)
    te_data = list(reader)
    te_data = np.asarray(te_data,dtype=np.int32)
    te_input = te_data[:,0:10]
    te_target = te_data[:,10:]
    fp.close()
    return te_input,te_target

def knn(item,item_set,k):
    """
    item为一个向量，item_set为向量列表，
    k近邻返回的是item_set的向量索引，也就是
    item的实际编号
    """    
    itemNum = len(item_set)
    dists = np.zeros([itemNum]) #记录和每一个item的距离
    for i,v in enumerate(item_set):
        dists[i] = ((item-v)**2).sum()

    #根据距离数组中的值排序数组索引,找出k近邻
    sortedDistIndices = dists.argsort() 

    return sortedDistIndices[:k]


def evaluate(pred_res,target,item_latent_vec):
    """
    预测结果是隐向量形式，target是编号形式
    预测结果以矩阵形式保存，每行对应一个用户，顺序
    与目标文件一致
    以k近邻来选择推荐
    n_user为用户数量
    """
    recall = 0.0
    n_user = len(pred_res)
    #预测目标的个数
    n_target= len(target[0])
    for k,v in enumerate(pred_res):
        #返回的推荐为用户编号
        recommed = knn(v,item_latent_vec,k)
        hit = 0
        for i in recommed:
            if i in target[k]:
                hit += 1
        recall += hit/n_target
    recall = recall/n_user 
    return recall
            
def main():

    #这里tr_data按每个用户一个list作为一个训练batch，数据表示为
    #item编号，还没有表示为隐向量
    tr_data = getTrainData()
    item_latent_vec = getMFData()
    
    #设置LSTM模型的参数
    #tr_data[0]为一个batch,tr_data[0][0]为第一个batch中第一个序列的长度，包括用户编号
    n_step = len(tr_data[0][0])-2 #最后一个留作训练目标
    
    #这里循环神经网络隐单元的数量与物品隐向量设置为相同(也可不同)
    hidden_size = item_latent_vec.shape[1]

    #这里用户数量，为了在模型中基于用户的偏置list的大小
    n_user = len(tr_data)

    model = LSTM(n_step=n_step,hidden_size=hidden_size,n_user=n_user)
    
    #训练轮数
    epoch = 10
    learning_rate = 0.05

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #在一个session内完成训练与预测
    with tf.Session() as sess:
        model.train(sess,tr_data,item_latent_vec,optimizer,epoch)
        ##预测结果以字典保存，关键字为用户编号
        
        te_input,te_target = getTestData()
        
        pred_res = model.pred(sess,te_input,item_latent_vec)

        recall = evaluate(pred_res,te_target,item_latent_vec)
        print "recall is %f"%recall


if __name__ == "__main__":
    #getmfdata()
    #gettestdata()
    main() 
