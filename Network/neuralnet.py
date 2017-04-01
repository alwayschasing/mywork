#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

class NeuralNetwork(object):
    
    """
    使用前馈神经网络，输入为一个序列的长度，输出为一个电影的概率
    """
    
    def __init__(self,seqlen,onehot_size,hidden_dims):
        self.seqlen = seqlen
        self.onehot_size = onehot_size
        """
        X_input:[batch_size,seqlen,onehot_size]
        Y_tar:[batch_size,onehot_size]
        """ 
        self.X_input = tf.placeholder(tf.float32,[None,seqlen,onehot_size],name="input")
        self.Y_tar = tf.placeholder(tf.float32,[None,onehot_size],name="target")

        batch_size = tf.shape(self.X_input)[0]
        #转换为[batch_size,seqlen*onehot_size]的输入
        cur_input = tf.reshape(self.X_input,[-1,seqlen*onehot_size])

        weights = []
        bias = []

        n_input = int(cur_input.get_shape()[1])
        """
        逐层设置系数与偏置
        """
        for layer_i,n_output in enumerate(hidden_dims):
            W = tf.Variable(tf.random_uniform(
                [n_input,n_output],
                -1.0/n_input,
                1.0/n_input,
                dtype=tf.float32
                ))
            b = tf.Variable(tf.zeros([n_output]))
            n_input = n_output
            weights.append(W)
            bias.append(b)

        outlayer_W = tf.Variable(tf.random_uniform(
            [hidden_dims[-1],onehot_size],
            -1.0/hidden_dims[-1],
            1.0/hidden_dims[-1]
            ))
        outlayer_b = tf.Variable(tf.zeros([onehot_size]))

        """
        逐层进行神经网络计算
        """
        for layer_i,n_output in enumerate(hidden_dims):
            cur_output = tf.nn.relu(tf.add(tf.matmul(cur_input,weights[layer_i]),b[layer_i]))
            cur_input = cur_output
        
        outs = tf.nn.relu(tf.matmul(cur_input,outlayer_W+outlayer_b))

        #logits:[batch_size,seqlen*onehot_size]
        logits = tf.reshape(outs,[batch_size,onehot_size])
        self.softmax_outs = tf.nn.softmax(logits,dim=-1)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits,self.Y_tar,dim=-1,name="loss")

            
    def train(self,sess,optimizer,n_epoch,train_input,train_target):

        sess.run(optimizer,feed_dict={self.X_input:train_input,self.Y_tar:train_target})


    def pred(self,sess,input):
        pred_res = sess.run(self.softmax_outs,feed_dict={self.X_input:input})
        return pred_res

if __name__ == "__main__":
    NeuralNetwork(9,3953,[2000,1000,400])
