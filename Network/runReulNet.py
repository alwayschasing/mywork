#!/usr/bin/env python
# coding=utf-8

import csv
import tensorflow as tf
import numpy as np
from neuralnet import NeuralNetwork
import time
"""
不分用户,使用全部序列数据,以前9个电影预测第10个电影
"""

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
    #首项表示用户编号
    for line in lines:
        training_data.append(line[1:])
    fpin.close()
    training_data = np.asarray(training_data,np.int32)
    print training_data.shape
    #training_data:[n_user,var_batch_size,n_step]
    return training_data

def getTestData():
    fp = open("/home/lrh/graduation_project/data/ml-1m/rnntestdata.csv","r")
    reader = csv.reader(fp)
    te_data = list(reader)
    te_data = np.asarray(te_data,dtype=np.int32)
    te_input = te_data[:,1:10]
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
    hidden_dims = [300,200,100,50]

    #参数有:n_step,hidden_size,item_code_size,u_code_size,latent_vec_size
    model = NeuralNetwork(seqlen,onehot_size,hidden_dims)
    
    #训练轮数
    n_epoch = 15
    learning_rate = 0.1
    #train_input = tr_data[:,:-1]
    #train_target = tr_data[:,-1]
    #batch_size = train_input.shape[0]
    batch_size = 10000
    total_size = tr_data.shape[0]

    n_batch = total_size/batch_size


    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    begintrain = time.time()
    print "start train"
    #在一个session内完成训练与预测
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        opt = optimizer.minimize(model.loss)
        for epoch in xrange(n_epoch):
            """
            进行n_epoch轮训练,每轮训练是逐batch进行的
            """
            n = 0
            for batch in xrange(n_batch):
                onehot_input = np.zeros([batch_size,seqlen,onehot_size])
                onehot_target = np.zeros([batch_size,onehot_size])
                
                n = 100*batch
                #生成onehot编码的输入与输出
                for i in range(batch_size):
                    onehot_target[i][tr_data[n+i][-1]] = 1
                    for j in range(seqlen):
                        onehot_input[i][j][tr_data[n+i][j]] = 1

                _,loss = sess.run([opt,model.loss],feed_dict={model.X_input:onehot_input,model.Y_tar:onehot_target})

            #训练最后一个batch为100的数据,数据设置的时候倒着来
            onehot_input = np.zeros([batch_size,seqlen,onehot_size])
            onehot_target = np.zeros([batch_size,onehot_size])
            for i in xrange(batch_size):
                onehot_target[-i-1][tr_data[-i-1][-1]] = 1
                for j in xrange(seqlen):
                    onehot_input[-i-1][j][tr_data[-i-1][j]] = 1
            _,loss = sess.run([opt,model.loss],feed_dict={model.X_input:onehot_input,model.Y_tar:onehot_target})
            aveloss = loss.mean()
            sumloss = loss.sum()
            print "epoch %d cost is %f"%(epoch,sumloss)

        endtrain = time.time()
        print "train run %d miniutes"%((endtrain-begintrain)/60)
        print "start run pred"

        """
        开始预测
        """
 
        te_input,te_target = getTestData()
        n_te_lines = te_input.shape[0]
        #将te_input转换为onehot编码的形式
        _te_input = np.zeros([n_te_lines,seqlen,onehot_size])

        for i in xrange(n_te_lines):
            for j in xrange(seqlen):
                _te_input[i][j][te_input[i][j]] = 1

        """
        #预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为3953的向量 
        #向量每一项对应一部电影的概率值
        """   
        pred_res = model.pred(sess,_te_input)
        print "start evaluate"

        recall = evaluate(pred_res,te_target,seqlen)
        print "recall is %f"%recall



if __name__ == "__main__":
    start = time.time()
    #getTestData()
    #getTrainData()
    main() 
    end = time.time()
    print "has run %d miniutes"%((end-start)/60)
