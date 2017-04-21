#!/usr/bin/env python
# coding=utf-8

import csv
import tensorflow as tf
import numpy as np
import time
from onehotRNN import NetworkModel 

rootdir = "/home/lrh/graduation_project/"
neighbor_k = 10


def getTrainData():
    """
    返回的数据：涉及同一用户的序列数据组成一个列表，即为一个batch
    ,不同的batch大小不同，所有列表组成返回数据，例如对于用户1：
    [batch_size]
    return [n_user,batch_size(不同用户大小不一样),vec_size]
    每个batch的每一行的行首为用户编号
    """
    fpin = open("/home/lrh/graduation_project/data/ml-1m/paperTraindata.csv","r")
    lines = list(csv.reader(fpin))
    training_data = np.asarray(lines,np.int32)
    print training_data.shape 
    return training_data

def getTestData():
    fp = open("/home/lrh/graduation_project/data/ml-1m/paperTestdata.csv","r")
    reader = csv.reader(fp)
    te_data = list(reader)
    te_data = np.asarray(te_data,dtype=np.int32)
    te_input = te_data[:,0:19]
    te_target = te_data[:,19:]
    fp.close()
    print te_input.shape
    print te_target.shape
    return te_input,te_target


def evaluate(pred_res,target,recommend_len):

    """
    #预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为max_item_index+1的向量 
    #向量每一项对应一部电影的概率值
    """   

    fp = open("/home/lrh/graduation_project/data/ml-1m/userbased.train.csv","r")
    userhistory = list(csv.reader(fp))
    recall = 0.0
    n_user = len(pred_res)
    #预测目标的个数
    n_target= len(target[0])
    savefp = open("/home/lrh/graduation_project/data/ml-1m/predicted_paper.csv","w")
    writer = csv.writer(savefp)
    hituser = 0
    for k,v in enumerate(pred_res):
        #返回的推荐为用户编号
        #该用户的历史列表，转换为整数型
        history = np.asarray(userhistory[k][1:],np.int32)
        recommend = v.argsort() 
        #rec_list = recommend[-recommend_len:]
        #过滤掉已经看过的电影
        rec_list = []
        count = 0
        for index in recommend[-1::-1]:
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
            hituser += 1
        recall += float(hit)/n_target

    print "hituser:%d"%hituser
    averec = recall/hituser
    print "relhit averec:%f"%averec
    recall = recall/n_user 
    savefp.close()
    return recall
            
def main():

    #这里tr_data还没有分batch
    tr_data = getTrainData()
    batch_size = 2000
    batch = tr_data.shape[0]/batch_size
    #设置LSTM模型的参数
    #tr_data[0]为一个batch,tr_data[0][0]为第一个batch中第一个序列的长度，包括用户编号
    #training_data:[n_user,var_batch_size,n_step]
    n_step = tr_data.shape[1]/2-1 
    
    #这里循环神经网络隐单元的大小
    hidden_size = 15
    max_item_index = 3952    
    max_user_index = 6040

    item_code_size = max_item_index+1
    u_code_size = max_user_index+1
    r_code_size = 5 
    beta = 0.2 #正则化系数

    #参数有:n_step,hidden_size,item_code_size,u_code_size,beta
    model = NetworkModel(n_step,hidden_size,item_code_size,u_code_size,r_code_size,beta)
    
    #训练轮数
    epoch = 6
    learning_rate = 0.1

    print "start train"
    begin = time.time()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.minimize(model.cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate)
    #在一个session内完成训练与预测
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for n_epoch in range(epoch):
            """
            按batch训练:
            """
            cost = 0.0
            for n_batch in range(batch):
                #生成批处理训练数据
                ba_trdata = tr_data[n_batch:n_batch+batch_size]
                batch_cost = model.train(sess,optimizer,ba_trdata,item_code_size,u_code_size,r_code_size)
                cost += batch_cost
            #训练最后一批
            ba_trdata = tr_data[-batch_size:]
            batch_cost = model.train(sess,optimizer,ba_trdata,item_code_size,u_code_size,r_code_size)
            cost += batch_cost

            print "epoch %d cost is %f"%(n_epoch,cost)
        ##预测结果以字典保存，关键字为用户编号
        
        now = time.time()
        print "training has run %d miniutes"%((now-begin)/60)
        te_input,te_target = getTestData()

        """
        #预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为max_item_index+1的向量 
        #向量每一项对应一部电影的概率值
        """   
        pred_res = model.pred(sess,te_input,item_code_size,u_code_size,r_code_size)

        save_rnnres(te_input,pred_res,20)
        #recommed_len = len(te_target[0])
        #recall = evaluate(pred_res,te_target,recommed_len)
        #print "recall is %f"%recall

def save_rnnres(te_input,pred_res,n_rec):
    #每个用户的历史信息
    hisfp = open("/home/lrh/graduation_project/data/ml-1m/userbased.train.csv","r") 
    his = list(csv.reader(hisfp))

    fp = open("/home/lrh/graduation_project/data/ml-1m/rnn_rec_res.csv","w")
    #writer = csv.writer(fp)
    n = te_input.shape[0]
    for i in xrange(n):
        #每个用户有n_rec个推荐
        u = te_input[i][0]
        history = np.asarray(his[i][1:],np.int32)
        count = 0
        tmp_list = []
        rec_list = pred_res[i].argsort()
        for item in rec_list[-1::-1]:
            if item not in history:
                tmp_list.append(item)
                count += 1
            if count >= n_rec:
                break
        for item in tmp_list:
            fp.write(str(u)+" "+str(item)+"\n")


def rank_recommend():
    pass




if __name__ == "__main__":
    start = time.time()
    #getTrainData()
    #getTestData()
    main() 
    end = time.time()
    print "total time %d"%((end-start)/60)
