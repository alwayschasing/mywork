#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import math

class LSTM(object):

    #搭建神经网络结构
    def __init__(self,n_step,hidden_size,n_user):
        
        self.n_step = n_step
        self.hidden_size = hidden_size

        self.x_input = tf.placeholder(tf.float32,[None,n_step,hidden_size],name="x_input")
        self.y_target = tf.placeholder(tf.float32,[None,n_step,hidden_size],name="y_target")
        batch_size = tf.shape(self.x_input)[0]
        #将输入数据变换为rnn网络接受的形式
        inputs = tf.transpose(self.x_input,[1,0,2])
        inputs = tf.reshape(inputs,[-1,hidden_size])
        inputs = tf.split(0,n_step,inputs)
        #这里rnn的hidden_size与输入数据的大小相同 
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True)
        self.rnn_outputs,self.states = tf.nn.rnn(lstm,inputs,dtype=tf.float32)
        
        inner_outputs = tf.pack(self.rnn_outputs)
        #转换成[batch_size,n_step,hidden_size]的数据形式
        inner_outputs = tf.transpose(inner_outputs,[1,0,2])
        #inner_outputs = tf.reshape(inner_outputs,[-1,hidden_size])
        #now outputs shape is [batch_size*n_step,hidden_size]

        #定义输出层的参数以及用户偏置

        #w = tf.Variable( tf.random_uniform([hidden_size,hidden_size],
                            #-1.0/math.sqrt(hidden_size),
                             #1.0/math.sqrt(hidden_size)))

        #bias = tf.Variable(tf.zeros([n_user,batch_size*n_step,hidden_size],dtype=tf.float32),name="userBias")

        #self.u = tf.placeholder(tf.int32,[1],name="user_index")
        #outputs = tf.add(tf.matmul(self.rnn_outputs,w)+bias[self.u])

        #恢复与输入相同的形式
        self.outputs = tf.reshape(inner_outputs,[batch_size,n_step,hidden_size])
        
        ##使用均方误差
        #self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.outputs,self.y_target),2))
        self.cost = tf.nn.l2_loss(tf.subtract(self.outputs,self.y_target))

    def train(self,sess,train_data,i_latent_set,optimizer,epoch):
        #i_latent_set表示物品隐向量表示集合
        sess.run(tf.global_variables_initializer())
        optimizer = optimizer.minimize(self.cost)
        #batch的数量，这里一个用户的数据为一个batch
        n_batch = len(train_data)
        for k in range(epoch):
            cost = 0
            for i in range(n_batch):
                #user = train_data[i][0][0]
                #生成一个batch的训练数据
                batch_input = []
                batch_target = []
                #每个序列数据为一个line
                for line in train_data[i]:
                    tmp_input =[]
                    tmp_target = []
                    for i_index in line[1:-1]:
                        i_index = int(i_index)
                        tmp_input.append(i_latent_set[i_index])
                    for i_index in line[2:]:
                        i_index = int(i_index)
                        tmp_target.append(i_latent_set[i_index])

                    batch_input.append(tmp_input) 
                    batch_target.append(tmp_target)

                #,生成一个batch的数据后训练,数据序列长度为9
                sess.run(optimizer,feed_dict={
                    self.x_input:batch_input,
                    self.y_target:batch_target})
                #分批累计平均损失,这里train_data[i].shape[0]指一个
                #用户训练序列的个数
                cost += sess.run(self.cost,feed_dict={
                    self.x_input:batch_input,
                    self.y_target:batch_target})/len(train_data[i])
            print "the %d epoch cost is %f"%(k,cost/n_batch)

    def pred(self,sess,te_data,item_latent_vec):

        #使用字典保存预测结果，关键字为用户编号
        res = dict()
        input = []
        for line in te_data:
            #u = line[0]
            te_input = line[1:]
            tmp = []
            for i in te_input: 
                tmp.append(item_latent_vec[int(i)])
            input.append(tmp)
            
        outputs = sess.run(self.outputs,feed_dict={
            self.x_input:input,
            #self.u:u
        })
        res = outputs[:,-1,:]
        return res

if __name__ == "__main__":

    model = LSTM(n_step=9,hidden_size=10,n_user=6040)
    #test
