#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

class LstmModel(object):
    
    def __init__(self,batch_size,n_step,hidden_size,lr=0.05):
        #input_size is (batch_size,n_step,n_input)
        #here is batch_size is the number of all sequences
        self.batch_size = batch_size
        self.n_step = n_step
        self.hidden_size = hidden_size
        self.lr = lr

        self.X = tf.placeholder(tf.float32,[None,n_step,hidden_size],name="X_inputs")
        self.Y = tf.placeholder(tf.float32,[None,n_step,hidden_size],name="targets")

        X = tf.transpose(self.X,[1,0,2]) #permute batch_size and time_step
        XR = tf.reshape(X,[-1,hidden_size])
        #XR shape:(time_step*batch_size,hidden_size)
        X_split = tf.split(0,n_step,XR) #split them to n_step size
        #each array shape:(batch_size,hidden_size)

        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True)
        #if need multiple layer of lastm
        #cell = tf.contrib.rnn.MultiRNNCell([lstm]*num_layer,state_is_tuple=True)
        self.outputs,self.states = tf.nn.rnn(lstm, X_split,dtype=tf.float32)

        #losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example()
        self.outputs = tf.reshape(tf.concat_v2(self.outputs, 1), [-1, hidden_size])
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits = [tf.reshape(self.outputs,[-1])],
            targets = [tf.reshape(self.Y,[-1])], 
            weights = [tf.ones([self.batch_size*self.n_step*hidden_size],dtype=tf.float32)],
            average_across_timesteps = True, 
            softmax_loss_function = self.ms_error,
            name = 'losses'
            ) 
        self.cost = tf.div(tf.reduce_sum(losses,name='losses_sum'),self.batch_size,name='avgcost')
        self.sess = tf.Session()
        self.init = tf.initialize_all_variables()
    
    def ms_error(self,y_pre,y):
        return tf.square(tf.sub(y_pre,y))
    def train(self,x,y,training_epoch):
        epoch = 0
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)
        self.sess.run(self.init)
        while epoch < training_epoch:
            self.sess.run(optimizer,feed_dict={self.X:x,self.Y:y})
            epoch += 1
            print "the %d epoch cost is %f"%(epoch,self.cost)
    def pred(self,testinput):
        return self.sess.run(self.outputs,feed_dict={self.X:testinput})


         


