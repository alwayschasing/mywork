#/usr/bin/env python
# coding=utf-8

import numpy as np
import csv
"""
实验对比的baseline
"""

def get_P_Q_matrix():
    fp = open("/home/lrh/graduation_project/lastfm/MFModel","r")
    lines = fp.readlines()
    n_line = len(lines)
    max_n_user = 0
    max_n_item = 0
    for i in range(5,n_line):
        line = lines[i].split()
        flag = line[0][0]
        if flag == 'p':
            index = int(line[0][1:])
            if index > max_n_user:
                max_n_user = index
        if flag == 'q':
            index = int(line[0][1:])
            if index > max_n_item:
                max_n_item = index
    
    print "max_n_user:%d"%max_n_user
    print "max_n_item:%d"%max_n_item

    p_matrix = np.zeros([max_n_user+1,10],dtype=np.float32)
    q_matrix = np.zeros([max_n_item+1,10],dtype=np.float32)

    for i in range(5,n_line):
        line = lines[i].split()
        flag = line[0][0]
        if flag == 'p':
            index = int(line[0][1:])
            sign = line[1]
            if(sign == 'T'):
                vec = line[2:]
                p_matrix[index] = vec
        elif flag == 'q':
            index = int(line[0][1:])
            sign = line[1]
            if(sign == 'T'):
                vec = line[2:]
                q_matrix[index] = vec
    print "get the P,Q matrix"
    return p_matrix,q_matrix

def getTestData():
    fp = open("/home/lrh/graduation_project/data/lastmf/finUserBasedTest.csv","r")
    reader = csv.reader(fp)
    data = list(reader)
    data = np.asarray(data,dtype=np.int32)
    """
    test data 每一行为一个用户的测试数据,行首为用户编号,待预测列表大小为10
    """
    print "get Test Data"
    return data

def predict(users,p_matrix,q_matrix):
    
    #n_user = users.shape[0]
    u_vecs = p_matrix[users] #shape[6040,10]
    u_mat = np.asmatrix(u_vecs)
    q_mat = np.asmatrix(q_matrix)

    rating_pred = u_mat*q_mat.T

    rating_pred = np.asarray(rating_pred)

    pred_items = []

    """
    去掉用户历史记录中的物品
    """
    h_fp = open("/home/lrh/graduation_project/data/lastmf/finUserBasedTrain.csv","r")
    his = list(csv.reader(h_fp))


    for k,one_user_pred in enumerate(rating_pred):
        history = np.asarray(his[k][1:],np.int32)
        recommend = one_user_pred.argsort()
        count = 0
        rec_list = []
        for index in recommend[-1::-1]:
            if index not in history:
                rec_list.append(index)
                count += 1
                if count >= 10:
                    break
        pred_items.append(rec_list)

    pred_items = np.asarray(pred_items,dtype=np.int32)
    return pred_items

def evaluate(te_data,pred_res):
    if te_data.shape[0] != pred_res.shape[0]:
        print "data shape get error"
        return 0
    
    n_user = te_data.shape[0]

    recall = 0.0
    hit_user = 0
    for k,v in enumerate(pred_res):
        hit = 0
        for i in v:
            if i in te_data[k]:
                hit += 1
        if hit != 0:
            hit_user += 1
        recall += float(hit)/10

    print "hit user:%d"%hit_user

    recall = recall/n_user

    return recall

     

def main():
    p_matrix,q_matrix = get_P_Q_matrix()
    te_data = getTestData()
    users = te_data[:,0] #shape:[1,480]
    predict_res = predict(users,p_matrix,q_matrix)

    te_list = te_data[:,1:]
    recall = evaluate(te_list,predict_res)
    print "recall is %f"%recall
    
if __name__=="__main__":
    #get_P_Q_matrix()
    #getTestData()
    main()
