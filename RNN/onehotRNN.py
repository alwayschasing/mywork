#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import math

class LSTM(object):
    """
    这里的模型只是对item的输入用lstm建模，最后将lstm的输出与user的onehot编码变换进行结合
    """

    #搭建神经网络结构,部分为lstm
    def __init__(self,n_step,hidden_size,n_user):
        
        self.n_step = n_step
        self.hidden_size = hidden_size
        #这里的输入为电影onehot编码,电影隐向量
        self.item = tf.placeholder(tf.float32,[None,n_step,item_size],name="item")
        self.i_latent_vec = tf.placeholder(tf.float32,[None,n_step,i_vec_size],name="i_latent_vec")

        #变换数据为可矩阵相乘,变换后item形式为：[n_step*batch_size,item_size]
        var_item = tf.transpose(self.item,[1,0,2])
        var_item = tf.reshape(var_item,[-1,item_size])

        #同样的变换应用与i_latent_vec
        var_i_vec = tf.transpose(self.i_latent_vec,[1,0,2])
        var_i_vec = tf.reshape(var_i_vec,[-1,i_vec_size])

        #定义变换矩阵V[item_size,hidden_size]，W[i_latent_vec_size,hidden_size]
        V = tf.Variable(tf.random_uniform(
            [item_size,hidden_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="V"
            ))

        W = tf.Variable(tf.random_uniform(
            [i_latent_vec_size,hidden_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="W"
            ))

        #产生lstm的输入[n_step*batch_size,hidden_size]
        inputs = tf.add(tf.matmul(self.item,V),tf.matmul(self.i_latent_vec))
        inputs = tf.split(0,n_step,inputs) #分step批同时处理

        #batch_size = tf.shape(self.x_input)[0]
        ##将输入数据变换为rnn网络接受的形式
        #inputs = tf.transpose(self.x_input,[1,0,2])
        #inputs = tf.reshape(inputs,[-1,hidden_size])
        #inputs = tf.split(0,n_step,inputs)
        #这里rnn的hidden_size与输入数据的大小相同 
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True)
        self.rnn_outputs,self.states = tf.nn.rnn(lstm,inputs,dtype=tf.float32)
        
        #now shape is [n_step,batch_size,hidden_size]
        inner_outputs = tf.pack(self.rnn_outputs) 
        inner_outputs = tf.reshape(inner_outputs,[-1,hidden_size])
        ##转换成[batch_size,n_step,hidden_size]的数据形式
        #inner_outputs = tf.transpose(inner_outputs,[1,0,2])

        #与lstm输出相乘的权值Y
        Y = tf.Variable(tf.random_uniform(
            [hidden_size,item_size]
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="Y"
            ))

        """
        定义用户部分的模型
        """
        self.user = tf.placeholder(tf.float32,[None,u_code_size],name="usercode")
        
        self.ulaten_vec = tf.placeholder(tf.float32,[None,vec_size],name="user_latent_vec")

        #定义用户模型部分参数P,Q
        
        P = tf.Variable(tf.random_uniform(
            
            ))

        Q = tf.Variable(tf.random_uniform(

            ))

        #用户模型部分的输出为user*P+ulaten_vec*Q
        u_model_out = tf.add(tf.matmul(self.user,P),tf.matmul(self.ulaten_vec,Q))

        #与用户模型输出相乘的矩阵Z
        Z = tf.Variable(tf.random_uniform(

            ))

        #得到最终输出O
        self.Out = 


        #目标向量
        self.y_target = tf.placeholder(tf.float32,[None,n_step,hidden_size],name="y_target")

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
                tmp.append(item_latent_vec[i])
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
