#!/usr/bin/env python
# coding=utf-8

import csv
import tensorflow as tf
import numpy as np
from neuralnet import NeuralNetwork
rootdir = "/home/lrh/graduation_project/"


def getTrainData():
    """
    返回的数据：涉及同一用户的序列数据组成一个列表，即为一个batch
    ,不同的batch大小不同，所有列表组成返回数据，例如对于用户1：
    [batch_size]
    return [n_user,batch_size(不同用户大小不一样),vec_size]
    每个batch的每一行的行首为用户编号
    """
    fpin = open("/home/lrh/graduation_project/data/ml-1m/rnndata2.csv","r")
    lines = list(csv.reader(fpin))
    training_data = []
    #表示用户编号
    for line in lines:
        training_data.append(line[1:])
    fpin.close()
    training_data = np.asarray(training_data,np.int32)
    #training_data:[n_user,var_batch_size,n_step]
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


def evaluate(pred_res,target,recommend_len):

    """
    #预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为max_item_index+1的向量 
    #向量每一项对应一部电影的概率值
    """   

    fp = open("../data/ml-1m/userbased.train.csv","r")
    userhistory = list(csv.reader(fp))
    recall = 0.0
    n_user = len(pred_res)
    #预测目标的个数
    n_target= len(target[0])
    savefp = open("../data/ml-1m/predicted_res.csv","w")
    writer = csv.writer(savefp)
    hituser = 0
    for k,v in enumerate(pred_res):
        #返回的推荐为用户编号
        #该用户的历史列表，转换为整数型
        history = np.asarray(userhistory[k][1:],np.int32)
        recommend = v.argsort() 
        #过滤掉已经看过的电影
        rec_list = []
        count = 0
        for index in recommend:
            if index not in history:
                rec_list.append(index)
                count += 1
                if count >= recommend_len:
                    break
        
        #统计每个用户的命中数
        writer.writerow(rec_list)
        hit = 0
        for i in rec_list:
            if i in target[k]:
                hit += 1
        if hit != 0:
            print hit
            hituser += 1
        recall += float(hit)/n_target

    print "hituser:%d"%hituser
    averec = recall/hituser
    print "relhit averec:%f"%averec
    recall = recall/n_user 
    savefp.close()
    return recall
            
def main():

    #这里tr_data:each row is a training sequence数据表示为
    #item编号
    #training_data:[batch_size,seqlen+1,onehot_size]
    tr_data = getTrainData()
    
    

    seqlen = 9
    onehot_size = 3953
    hidden_dims = []

    #参数有:n_step,hidden_size,item_code_size,u_code_size,latent_vec_size
    model = NeuralNetwork(seqlen,onehot_size,hidden_dims)
    
    #训练轮数
    epoch = 10
    learning_rate = 0.05
    train_input = tr_data[:,:-1]
    train_target = tr_data[:,-1]
    batch_size = train_input.shape[0]
    onehot_input = np.zeros([batch_size,seqlen,onehot_size])
    onehot_target = np.zeros([batch_size,onehot_size])

    #生成onehot编码的输入与输出
    for i in range(batch_size):
        onehot_target[i][train_target[i][0]] = 1
        for j in range(seqlen):
            onehot_input[i][train_input[i][j]] = 1

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #在一个session内完成训练与预测
    with tf.Session() as sess:

        model.train(sess,optimizer,epoch,train_input,train_target)
        ##预测结果以字典保存，关键字为用户编号
        
        te_input,te_target = getTestData()

        """
        #预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为3953的向量 
        #向量每一项对应一部电影的概率值
        """   
        pred_res = model.pred(sess,te_input)

        recall = evaluate(pred_res,te_target)
        print "recall is %f"%recall



if __name__ == "__main__":
    #getMFData()
    #gettestdata()
    main() 
