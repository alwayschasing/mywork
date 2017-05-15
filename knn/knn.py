#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cPickle as pickle
import csv
import time


n_user = 6041
n_item = 3953

def getTrainData():
    fp = open("/home/lrh/graduation_project/data/ml-1m/userbased.trainMF.csv","r") 

    R_matrix = np.zeros([n_user,n_item],dtype=np.float32)

    lines = list(csv.reader(fp))

    for line in lines:
        user = int(line[0])
        n = len(line)/2
        for i in range(0,n):
            item = int(line[2*i+1])
            rating = int(line[2*i+2])
            R_matrix[user][item] = rating

    fp.close()
    print "get rating matrix"
    print R_matrix.shape
    return R_matrix

def getUserSimilarity():
    rating_matrix = getTrainData()
    userSimilarity = np.zeros([n_user,n_user],dtype=np.float32)

    avgs = rating_matrix.mean(axis=1,dtype=np.float32)
    for u in range(1,n_user):
        for v in range(u+1,n_user):
            #皮尔逊相似度
            v1 = rating_matrix[u] - avgs[u]
            v2 = rating_matrix[v] - avgs[v]
            dist = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            userSimilarity[u][v] = dist

    for u in range(1,n_user):
        for v in range(1,u):
            userSimilarity[u][v] = userSimilarity[v][u]

    print userSimilarity.shape
    save = open("/home/lrh/graduation_project/KNN/userSimilarity","wb")
    pickle.dump(userSimilarity,save)
    #return userSimilarity


def userBasedKnn(kN):

    rating_matrix = getTrainData() #获得训练数据的评分矩阵
    #userSimilarities = getUserSimilarity(rating_matrix) #得到用户相似度矩阵
    userSimilarities = pickle.load(open("/home/lrh/graduation_project/KNN/userSimilarity","rb"))

    #保存每个用户未知评分的预测结果,顺便剔除了用户看过的电影
    pred_res = np.zeros([n_user,n_item],dtype=np.float32)

    for user in range(1,n_user):
        #依次对每个用户进行处理
        for item in range(1,n_item):
            if rating_matrix[user][item] == 0.0:
                #获得对目标电影评过分且距离最近的k个用户
                neighbors = userSimilarities[user].argsort()
                neighbor_list = []
                k = 0
                for index in neighbors:
                    if rating_matrix[index][item] != 0:
                        neighbor_list.append(index)
                        k += 1
                        if k >= kN:
                            break
                if k < kN:
                    #print "has no k neighbors"
                    #无法获得k个近领,跳过该电影预测
                    continue

                denominator = 0.0
                for neig in neighbor_list:
                    denominator += np.fabs(userSimilarities[user][neig])
                numerator = 0.0
                for neig in neighbor_list:
                    numerator += userSimilarities[user][neig]*rating_matrix[neig][item]
                pred = numerator/denominator

                pred_res[user][item] = pred
    save = open("/home/lrh/graduation_project/KNN/pred_res","wb")
    pickle.dump(pred_res,save)
    return pred_res      


def getItemSimilarity(rating_matrix):
    pass
                
def itemBasedKnn(kN):
    pass
    
def getTestData():
    fp = open("/home/lrh/graduation_project/data/ml-1m/userbased.test.csv","r")
    te_data = list(csv.reader(fp))
    te_data = np.asarray(te_data,dtype=np.int32)
    return te_data

def evalRecall(te_data,pred_res):
    
    hituser = 0
    recall = 0.0
    n_rec = te_data.shape[1]-1
    for line in te_data:
        hit = 0
        user = line[0]
        recommend = pred_res[user].argsort()[-10:]
        for item in line[1:]:
            if item in recommend:
                hit += 1
        if hit > 0:
            recall += float(hit)/n_rec
            hituser += 1

    recall = recall/(te_data.shape[0]-1)
    print "hithuser %d"%hituser
    return recall

def main():
    #kN = 5
    #pred_res = userBasedKnn(kN)

    te_data = getTestData()
    pred_res = pickle.load(open("/home/lrh/graduation_project/data/knn/pred_res","rb"))
    recall = evalRecall(te_data,pred_res)
    print "recall is %f"%recall
    

if __name__ == "__main__":
    start = time.time()
    #getTrainData() 
    #getUserSilimarity()
    #getUserSimilarity()
    #userBasedKnn(5)
    main()
    end = time.time()
    print "run %d minautes"%((end-start)/60)

    
