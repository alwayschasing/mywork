#!/usr/bin/env python
# coding=utf-8

import csv
import tensorflow as tf
import numpy as np
from onehotRNN import NetworkModel 

rootdir = "/home/lrh/graduation_project/"
neighbor_k = 10

def getMFData():
    fpin = open("/home/lrh/graduation_project/MF/MFModel","r")
    lines = fpin.readlines()
    num = len(lines)
    
    max_n_item = 0
    max_n_user = 0
    for i in range(5,num):
        line = lines[i].split()
        flag = line[0][0]
        if flag == 'p':
            u_index = int(line[0][1:])
            if u_index > max_n_user:
                max_n_user = u_index
        elif flag == 'q':
            i_index = int(line[0][1:])
            if i_index > max_n_item:
                max_n_item = i_index

    latent_dim = 10
    #shape[0]加1是因为实际数据从1而不是从0开始
    item_latent_vec = np.zeros([max_n_item+1,latent_dim],dtype=np.float32)
    user_latent_vec = np.zeros([max_n_user+1,latent_dim],dtype=np.float32)
    for i in range(5,num):
        line = lines[i].split()
        flag = line[0][0]
        if flag == 'p':
            u_index = int(line[0][1:])
            sign = line[1]
            if(sign == 'T'):
                vec = line[2:]
                user_latent_vec[u_index] = vec
        elif flag == 'q':
            i_index = int(line[0][1:])
            sign = line[1]
            if(sign=='T'):
                vec = line[2:]
                item_latent_vec[i_index] = vec
    print "has got MFdata"

    fpin.close()
    return user_latent_vec,item_latent_vec

def getTrainData():
    """
    返回的数据：涉及同一用户的序列数据组成一个列表，即为一个batch
    ,不同的batch大小不同，所有列表组成返回数据，例如对于用户1：
    [batch_size]
    return [n_user,batch_size(不同用户大小不一样),vec_size]
    每个batch的每一行的行首为用户编号
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

    #这里tr_data按每个用户一个list作为一个训练batch，数据表示为
    #item编号，还没有表示为隐向量
    tr_data = getTrainData()
    user_latent_vec,item_latent_vec = getMFData()
    
    #设置LSTM模型的参数
    #tr_data[0]为一个batch,tr_data[0][0]为第一个batch中第一个序列的长度，包括用户编号
    #training_data:[n_user,var_batch_size,n_step]
    n_step = len(tr_data[0][0])-2 #最后一个留作训练目标
    
    #这里循环神经网络隐单元的大小
    hidden_size = 20
    latent_vec_size = user_latent_vec.shape[1]
    max_item_index = 3952    
    max_user_index = 6040

    item_code_size = max_item_index+1
    u_code_size = max_user_index+1

    #参数有:n_step,hidden_size,item_code_size,u_code_size,latent_vec_size
    model = NetworkModel(n_step,hidden_size,item_code_size,u_code_size,latent_vec_size)
    
    #训练轮数
    epoch = 10
    learning_rate = 0.1

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #在一个session内完成训练与预测
    with tf.Session() as sess:

        model.train(sess,optimizer,epoch,tr_data,item_latent_vec,user_latent_vec,max_item_index,max_user_index)
        ##预测结果以字典保存，关键字为用户编号
        
        te_input,te_target = getTestData()

        """
        #预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为max_item_index+1的向量 
        #向量每一项对应一部电影的概率值
        """   
        pred_res = model.pred(sess,te_input,item_latent_vec,user_latent_vec,max_item_index,max_user_index)

        recommed_len = len(te_target[0])
        recall = evaluate(pred_res,te_target,recommed_len)
        print "recall is %f"%recall



if __name__ == "__main__":
    #getMFData()
    #gettestdata()
    main() 
