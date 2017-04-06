#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

class LSTM(object):

    #搭建神经网络结构
    def __init__(self,n_step,lat_vec_size,hidden_size):
        
        self.n_step = n_step
        self.hidden_size = hidden_size

        self.x_input = tf.placeholder(tf.float32,[None,n_step,lat_vec_size],name="x_input")
        self.y_target = tf.placeholder(tf.float32,[None,n_step,lat_vec_size],name="y_target")
        batch_size = tf.shape(self.x_input)[0]
        #将输入数据变换为rnn网络接受的形式
        inputs = tf.transpose(self.x_input,[1,0,2])
        inputs = tf.reshape(inputs,[-1,lat_vec_size])
        """
        对原始输入进行非线性变换再提供给RNN操作
        """
        W = tf.Variable(tf.ones(
            [lat_vec_size,hidden_size],
            dtype=tf.float32
            ))
        
        #shape:[n_step*batch_size,lat_vec_size]
        #inputs = tf.sigmoid(tf.matmul(inputs,W))
        inputs = tf.matmul(inputs,W)

        inputs = tf.split(0,n_step,inputs)
        #这里rnn的hidden_size与输入数据的大小相同 
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=0.5,state_is_tuple=True)
        self.rnn_outputs,self.states = tf.nn.rnn(lstm,inputs,dtype=tf.float32)
        
        inner_outputs = tf.pack(self.rnn_outputs)
        inner_outputs = tf.reshape(inner_outputs,[-1,hidden_size])#[n_step*batch_size,hidden_size]
        #转换成[batch_size,n_step,hidden_size]的数据形式
        #inner_outputs = tf.transpose(inner_outputs,[1,0,2])
        

        """
        恢复与输出相同的形式
        """
        Z = tf.Variable(tf.ones(
            [hidden_size,lat_vec_size],
            dtype=tf.float32
            ))
        outs = tf.matmul(inner_outputs,Z)

        outs = tf.reshape(outs,[n_step,batch_size,lat_vec_size])
        self.outputs = tf.transpose(outs,[1,0,2])
        
        ##使用均方误差
        #self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.outputs,self.y_target),2))
        self.cost = tf.nn.l2_loss(tf.subtract(self.outputs,self.y_target))

    def train(self,sess,train_data,i_latent_set,optimizer,epoch):
        #i_latent_set表示物品隐向量表示集合
        optimizer = optimizer.minimize(self.cost)
        #batch的数量，这里一个用户的数据为一个batch
        n_batch = len(train_data)
        for k in range(epoch):
            cost = 0.0
            """
            每个用户为一个batch 
            """
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

    def batch_train(self,sess,tr_data,item_latent_vec,optimizer,epoch,batch_size):
        opt = optimizer.minimize(self.cost)
        n = tr_data.shape[0]
        n_batch = n/batch_size
        for k in xrange(epoch):
            cost = 0.0
            for _batch in xrange(n_batch):
                inputs = []
                targets = []
                i = _batch*batch_size
                #处理每个batch中的每一行数据
                for line in tr_data[i:i+batch_size]:
                    tmp_inputs = []
                    tmp_targets = []
                    for index in line[1:-1]:
                        tmp_inputs.append(item_latent_vec[index])
                    for index in line[2:]:
                        tmp_targets.append(item_latent_vec[index])
                    inputs.append(tmp_inputs)
                    targets.append(tmp_targets)

                
                _,bat_cost = sess.run([opt,self.cost],feed_dict={
                    self.x_input:inputs,
                    self.y_target:targets
                })
                cost += bat_cost
            """
            每轮训练的最后一个batch
            """
            inputs = []
            targets = []
            for seq in tr_data[-batch_size:]:
                tmp_inputs = []
                tmp_targets = []
                for l_index in seq[1:-1]:
                    tmp_inputs.append(item_latent_vec[l_index])
                for l_index in seq[2:]:
                    tmp_targets.append(item_latent_vec[l_index])
                inputs.append(tmp_inputs)
                targets.append(tmp_targets)
                
            _,lastbat_cost = sess.run([opt,self.cost],feed_dict={
                self.x_input:inputs,
                self.y_target:targets
            })
            cost += lastbat_cost
            
            print "epoch %d cost is %f"%(k,cost)


    def pred(self,sess,te_data,item_latent_vec):

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

    model = LSTM(n_step=9,lat_vec_size=10,hidden_size=5)
    #test
