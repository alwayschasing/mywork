#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

class NetworkModel(object):
    """
    这里的模型只是对item的输入用lstm建模，最后将lstm的输出与user的onehot编码变换进行结合 """ 
    #搭建神经网络结构,部分为lstm
    def __init__(self,n_step,hidden_size,item_code_size,u_code_size,r_code_size,beta):
        """
        参数
        n_step: rnn循环的步数
        hidden_size: rnn部分隐藏单元大小
        item_code_size: item编码向量的大小 
        u_code_size: 用户编码向量的大小
        beta: 正则化系数
        """
        
        self.n_step = n_step
        self.hidden_size = hidden_size
        self.item_code_size = item_code_size
        self.u_code_size = u_code_size
        self.r_code_size = r_code_size
        #这里的输入为电影onehot编码,电影隐向量
        self.item = tf.placeholder(tf.float32,[None,n_step,item_code_size],name="item")
        self.rating = tf.placeholder(tf.float32,[None,n_step,r_code_size],name="rating")

        batch_size = tf.shape(self.item)[0]

        #变换数据为可矩阵相乘,变换后item形式为：[n_step*batch_size,item_code_size]
        _item = tf.transpose(self.item,[1,0,2])
        _item = tf.reshape(_item,[-1,item_code_size])
        _rating = tf.transpose(self.rating,[1,0,2])
        _rating = tf.reshape(_rating,[-1,r_code_size])


        #定义变换矩阵V[item_code_size,hidden_size]，W[latent_vec_size,hidden_size]
        W = tf.Variable(tf.random_uniform(
            [item_code_size,hidden_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="W"
            ))

        V = tf.Variable(tf.random_uniform(
            [r_code_size,hidden_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32
            ))


        #产生lstm的输入[n_step*batch_size,hidden_size]
        inputs = tf.sigmoid(tf.add(tf.matmul(_item,W),tf.matmul(_rating,V)))

        inputs = tf.split(0,n_step,inputs) #分step批同时处理


        #batch_size = tf.shape(self.x_input)[0]
        ##将输入数据变换为rnn网络接受的形式
        #inputs = tf.transpose(self.x_input,[1,0,2])
        #inputs = tf.reshape(inputs,[-1,hidden_size])
        #inputs = tf.split(0,n_step,inputs)
        #这里rnn的hidden_size与输入数据的大小相同 
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=0.5,state_is_tuple=True)
        self.rnn_outputs,_states = tf.nn.rnn(lstm,inputs,dtype=tf.float32)
        
        #now shape is [n_step,batch_size,hidden_size]
        inner_outputs = tf.pack(self.rnn_outputs) 
        inner_outputs = tf.reshape(inner_outputs,[-1,hidden_size])
        ##转换成[batch_size,n_step,hidden_size]的数据形式
        #inner_outputs = tf.transpose(inner_outputs,[1,0,2])

        #与lstm输出相乘的权值Y
        Y = tf.Variable(tf.random_uniform(
            [hidden_size,item_code_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="Y"
            ))
        #shape:[n_step*batch_size,item_code_size]
        rnn_outs = tf.matmul(inner_outputs,Y)
        

        """
        定义用户部分的模型
        """
        #需要输入的数据
        self.user = tf.placeholder(tf.float32,[None,self.n_step,u_code_size],name="usercode")
        

        _user = tf.transpose(self.user,[1,0,2])
        _user = tf.reshape(_user,[-1,u_code_size])
        #定义用户模型部分参数P,Q
        #用户模型部分隐单元大小默认与rnn部分相同
        u_model_hidden_size = hidden_size 

        P = tf.Variable(tf.random_uniform(
            [u_code_size,u_model_hidden_size],
            -1.0/u_code_size,
            1.0/u_code_size
            ))



        #用户模型部分的输出为user*P+ulaten_vec*Q,shape:[batch_size*n_step,u_model_hidden_size]
        u_inner_outs = tf.matmul(_user,P)
        #加一个激活函数,并将输出复制n_step份
        u_inner_outs = tf.nn.sigmoid(u_inner_outs)
        

        #与用户模型输出相乘的矩阵Z
        Z = tf.Variable(tf.random_uniform(
            [u_model_hidden_size,item_code_size],
            -1.0/u_model_hidden_size,
            1.0/u_model_hidden_size
            ))
        #用户模型的最终输出，变换到[n_step*batch_size,item_code_size]
        u_model_outs = tf.matmul(u_inner_outs,Z)
        u_model_outs = tf.reshape(u_model_outs,[n_step,batch_size,item_code_size])

        #得到最终输出O,使用softmax多分类
        rnn_outs = tf.reshape(rnn_outs,[n_step,batch_size,item_code_size])

        logits = tf.add(u_model_outs,rnn_outs)
        #恢复与输入，目标向量相同的形式
        logits = tf.transpose(logits,[1,0,2])#shape:[batch_size,n_step,item_code_size]
        #softmax:dim指做softmax计算的维度，默认为-1，即最后一个维度
        #self.Outs = tf.nn.log_softmax(logits,dim=-1,name="softmax_outs")
        self.Outs = tf.nn.softmax(logits,dim=-1,name="softmax_outs")

        #目标向量
        self.y_target = tf.placeholder(tf.float32,[None,n_step,item_code_size],name="y_target")

        
        ##损失使用交叉熵,并使用正则抑制过拟合
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Outs,labels=self.y_target)
                                +beta*tf.nn.l2_loss(Y)
                                +beta*tf.nn.l2_loss(Z)
                                +beta*tf.nn.l2_loss(W)
                                +beta*tf.nn.l2_loss(V)
                                +beta*tf.nn.l2_loss(P))


    def train(self,sess,optimizer,train_data,item_code_size,u_code_size,r_code_size):
        """
        将一个batch的数据准备为onehot形式,并进行一个batch的训练
        每个batch的一行形式为:
        行首为用户编号,之后为电影及评分对
        """
        batch_size = train_data.shape[0]

        batch_u_code = np.zeros([batch_size,self.n_step,u_code_size])
        batch_item_code = np.zeros([batch_size,self.n_step,item_code_size])
        batch_r_code = np.zeros([batch_size,self.n_step,r_code_size])
        batch_target = np.zeros([batch_size,self.n_step,item_code_size])

        for i in range(batch_size):
            user = train_data[i][0]
            for j in range(self.n_step):
                item = train_data[i][2*j+1]
                nextitem = train_data[i][2*(j+1)+1]
                rating = train_data[i][1+2*j+1]
                batch_u_code[i][j][user] = 1
                batch_item_code[i][j][item] = 1
                batch_r_code[i][j][rating-1] = 1 #rating值比其onehot编码1位置大1
                batch_target[i][j][nextitem] = 1
        _,cost = sess.run([optimizer,self.cost],feed_dict={
            self.item:batch_item_code,
            self.user:batch_u_code,
            self.rating:batch_r_code,
            self.y_target:batch_target})
        return cost 


    def pred(self,sess,te_data,item_code_size,u_code_size,r_code_size):
        """
        预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为max_item_index+1的向量 
        向量每一项对应一部电影的概率值
        """

        batch_size = te_data.shape[0]
        #使用数组保存预测结果，索引为用户编号

        batch_item_code = np.zeros([batch_size,self.n_step,item_code_size])
        batch_u_code = np.zeros([batch_size,self.n_step,u_code_size])
        batch_r_code = np.zeros([batch_size,self.n_step,r_code_size])

        for i in range(batch_size):
            user = te_data[i][0]
            for j in range(self.n_step):
                #行首为用户编号，所以j+1
                item = te_data[i][1+j*2]
                rating = te_data[i][1+j*2+1]
                batch_item_code[i][j][item] = 1
                batch_u_code[i][j][user] = 1
                batch_r_code[i][j][rating-1] = 1
            
        
        pred_res = sess.run(self.Outs,feed_dict={
            self.item:batch_item_code,
            self.rating:batch_r_code,
            self.user:batch_u_code
        })

        #预测结果为[batch_size,item_onehot_size]
        pred_res = pred_res[:,-1,:]

        return pred_res

if __name__ == "__main__":

    model = NetworkModel(n_step=9,hidden_size=10,item_code_size=3953,u_code_size=6041,r_code_size=5,beta=0.05)
    #test
