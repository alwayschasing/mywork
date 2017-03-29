#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

class LSTM(object):
    """
    这里的模型只是对item的输入用lstm建模，最后将lstm的输出与user的onehot编码变换进行结合
    """

    #搭建神经网络结构,部分为lstm
    def __init__(self,n_step,hidden_size,item_code_size,latent_vec_size,u_code_size):
        """
        参数
        n_step: rnn循环的步数
        hidden_size: rnn部分隐藏单元大小
        item_code_size: item编码向量的大小
        latent_vec_size: 使用的电影或者用户隐因子向量
        
        u_code_size: 用户编码向量的大小
        """
        
        self.n_step = n_step
        self.hidden_size = hidden_size
        #这里的输入为电影onehot编码,电影隐向量
        self.item = tf.placeholder(tf.float32,[None,n_step,item_code_size],name="item")
        self.i_latent_vec = tf.placeholder(tf.float32,[None,n_step,latent_vec_size],name="i_latent_vec")

        batch_size = tf.shape(self.item)[0]

        #变换数据为可矩阵相乘,变换后item形式为：[n_step*batch_size,item_code_size]
        _item = tf.transpose(self.item,[1,0,2])
        _item = tf.reshape(_item,[-1,item_code_size])

        #同样的变换应用与i_latent_vec
        _i_latent_vec = tf.transpose(self.i_latent_vec,[1,0,2])
        _i_latent_vec = tf.reshape(_i_latent_vec,[-1,latent_vec_size])

        #定义变换矩阵V[item_code_size,hidden_size]，W[latent_vec_size,hidden_size]
        V = tf.Variable(tf.random_uniform(
            [item_code_size,hidden_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="V"
            ))

        W = tf.Variable(tf.random_uniform(
            [latent_vec_size,hidden_size],
            -1.0/hidden_size,
            1.0/hidden_size,
            dtype=tf.float32,
            name="W"
            ))

        #产生lstm的输入[n_step*batch_size,hidden_size]
        inputs = tf.add(tf.matmul(_item,V),tf.matmul(_i_latent_vec,W))
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
            [hidden_size,item_code_size]
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
        self.user = tf.placeholder(tf.float32,[None,u_code_size],name="usercode")
        
        self.ulaten_vec = tf.placeholder(tf.float32,[None,latent_vec_size],name="user_latent_vec")


        #定义用户模型部分参数P,Q
        #用户模型部分隐单元大小默认与rnn部分相同
        u_model_hidden_size = hidden_size 

        P = tf.Variable(tf.random_uniform(
            [u_code_size,u_model_hidden_size]   
            ))

        Q = tf.Variable(tf.random_uniform(
            [latent_vec_size,u_model_hidden_size]
            ))

        #用户模型部分的输出为user*P+ulaten_vec*Q,shape:[batch_size,u_model_hidden_size]
        u_inner_outs = tf.add(tf.matmul(self.user,P),tf.matmul(self.ulaten_vec,Q))
        #加一个激活函数,并将输出复制n_step份
        u_inner_outs = tf.sigmoid(u_inner_outs)
        

        #与用户模型输出相乘的矩阵Z
        Z = tf.Variable(tf.random_uniform(
            [u_model_hidden_size,item_code_size]
            ))
        #用户模型的最终输出，变换到[n_step*batch_size,item_code_size]
        u_model_outs = tf.matmul(u_inner_outs,Z)

        #得到最终输出O,使用softmax多分类
        tmp_outs = tf.reshape(rnn_outs,[n_step,batch_size,item_code_size])
        for i in range(n_step):
            tmp_outs[i] = tf.add(tmp_outs[i],u_model_outs)
        #恢复与输入，目标向量相同的形式
        logits = tf.transpose(tmp_outs,[1,0,2])#shape:[batch_size,n_step,item_code_size]
        #softmax:dim指做softmax计算的维度，默认为-1，即最后一个维度
        self.Outs = tf.softmax(logits,dim=-1,name="softmax_outs")

        #目标向量
        self.y_target = tf.placeholder(tf.float32,[None,n_step,hidden_size],name="y_target")

        
        ##损失使用交叉熵
        self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.Outs,labels=self.y_target,dim=-1,name="loss")

    def train(self,sess,train_data,i_latent_set,u_latent_set,optimizer,epoch):
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

    model = LSTM(n_step=9,hidden_size=10)
    #test
