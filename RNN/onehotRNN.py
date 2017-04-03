#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

class NetworkModel(object):
    """
    这里的模型只是对item的输入用lstm建模，最后将lstm的输出与user的onehot编码变换进行结合 """ 
    #搭建神经网络结构,部分为lstm
    def __init__(self,n_step,hidden_size,item_code_size,u_code_size,beta):
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
        #这里的输入为电影onehot编码,电影隐向量
        self.item = tf.placeholder(tf.float32,[None,n_step,item_code_size],name="item")

        batch_size = tf.shape(self.item)[0]

        #变换数据为可矩阵相乘,变换后item形式为：[n_step*batch_size,item_code_size]
        _item = tf.transpose(self.item,[1,0,2])
        _item = tf.reshape(_item,[-1,item_code_size])


        #定义变换矩阵V[item_code_size,hidden_size]，W[latent_vec_size,hidden_size]
        V = tf.Variable(tf.random_uniform(
            [item_code_size,hidden_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="V"
            ))

        inputs_bias = tf.Variable(tf.zeros([hidden_size]))

        #产生lstm的输入[n_step*batch_size,hidden_size]
        inputs = tf.sigmoid(tf.add(tf.matmul(_item,V),inputs_bias))

        inputs = tf.split(0,n_step,inputs) #分step批同时处理


        #batch_size = tf.shape(self.x_input)[0]
        ##将输入数据变换为rnn网络接受的形式
        #inputs = tf.transpose(self.x_input,[1,0,2])
        #inputs = tf.reshape(inputs,[-1,hidden_size])
        #inputs = tf.split(0,n_step,inputs)
        #这里rnn的hidden_size与输入数据的大小相同 
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True)
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

        u_bias = tf.Variable(tf.zeros([u_model_hidden_size]))


        #用户模型部分的输出为user*P+ulaten_vec*Q,shape:[batch_size*n_step,u_model_hidden_size]
        u_inner_outs = tf.add(tf.matmul(_user,P),u_bias)
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
                                +beta*tf.nn.l2_loss(Z))


    def train(self,sess,optimizer,epoch,train_data,item_code_size,u_code_size):
        """
        将数据的部分准备也放在了这里，train_data只包含所有用户的编号数据，
        一个用户的数据为一个batch，每个batch的每一个为一个序列，行首为用户编号

        同时要生成用户以及item的one-hot编码,使用max_item_index,max_user_index
        用户数据需要n_step份以喂给神经网络
        """
        optimizer = optimizer.minimize(self.cost)
        #batch的数量，这里一个用户的数据为一个batch
        n_batch = len(train_data)
        for k in range(epoch):
            """
            训练epoch轮
            """
            cost = 0
            for i in range(n_batch):
                """
                按batch轮着训练
                一个用户的数据为一个batch
                """
                batch_size = len(train_data[i])
                user = int(train_data[i][0][0])
                batch_u_code = np.zeros([batch_size,self.n_step,u_code_size])
                batch_u_code[:,:,user] = 1 #将user编号作为位置索引
                #生成shape:[batch_size,n_step,latent_vec_size]

                #生成一个batch的训练数据
                batch_input = []
                batch_target = []

                #每个序列数据为一个line
                for line in train_data[i]:
                    #对序列数据one-hot编码
                    input_code = np.zeros([self.n_step,item_code_size])
                    target_code = np.zeros([self.n_step,item_code_size])

                    count = 0 #计数step数据处理    
                    for i_index in line[1:-1]:
                        i_index = int(i_index)
                        input_code[count][i_index] = 1
                        count += 1

                    count = 0
                    for i_index in line[2:]:
                        i_index = int(i_index)
                        target_code[count][i_index] = 1
                        count += 1
                    batch_input.append(input_code) 
                    batch_target.append(target_code)

                #生成一个batch的数据后训练,数据序列长度为9
                #分批累计平均损失,这里train_data[i].shape[0]指一个
                #用户训练序列的个数
                _,tmpcost = sess.run([optimizer,self.cost],feed_dict={
                    self.item:batch_input,
                    self.user:batch_u_code,
                    self.y_target:batch_target})
                tmpcost = tmpcost.mean()
                cost += tmpcost/len(train_data[i])
            print "the %d epoch cost is %f"%(k,cost/n_batch)

    def pred(self,sess,te_data,item_code_size,u_code_size):
        """
        预测返回的是一个列表，每一项为一个用户的预测，预测结果为一个大小为max_item_index+1的向量 
        向量每一项对应一部电影的概率值
        """

        batch_size = len(te_data)
        #使用数组保存预测结果，索引为用户编号

        batch_item = np.zeros([batch_size,self.n_step,item_code_size])
        batch_u_code = np.zeros([batch_size,self.n_step,u_code_size])
        n_step = len(te_data[0])-1
        for i in range(batch_size):
            line = te_data[i]
            u = int(line[0])
            for j in range(n_step):
                #行首为用户编号，所以j+1
                item = int(te_data[i][j+1])
                batch_item[i][j][item] = 1
                batch_u_code[i][j][u] = 1
            
        
        pred_res = sess.run(self.Outs,feed_dict={
            self.item:batch_item,
            self.user:batch_u_code,
        })

        #预测结果为[batch_size,item_onehot_size]
        pred_res = pred_res[:,-1,:]

        return pred_res

if __name__ == "__main__":

    model = NetworkModel(n_step=9,hidden_size=10,item_code_size=3953,u_code_size=6041,beta=0.05)
    #test
