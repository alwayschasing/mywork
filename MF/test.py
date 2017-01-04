# -*- coding: utf-8 -*-
from numpy import *
import  numpy as np
import os
import sys, idlelib.PyShell; idlelib.PyShell.warning_stream = sys.stderr

os.chdir("E:\\data")


#def loadExData():
#数据预处理:将ratings数据集变成矩阵形式
def dataProcess():
    juzhen = zeros([943, 1682],dtype='float64')
    fr = open("u.data")
    for line in fr.readlines():
        line = line.strip().split()
        juzhen[int(line[0]) - 1][int(line[1]) - 1] = int(line[2])
    np.savetxt('juzhen.csv', juzhen, delimiter=',')
    print "矩阵已建好"
    '''
    #np.savetxt('juzhen.csv', juzhen, delimiter=',')
    U, sigma, V = linalg.svd(juzhen)

    fr = open("sigma.txt",'a')
    for i in sigma:
       print >> fr, i
    fr.close()
    '''
    return juzhen

def matrix_fraction(juzhen, k, steps, alpha, beta):
    m = juzhen.shape[0]#用户个数
    n = juzhen.shape[1]#物品个数
    P = zeros([m,k],dtype='float64')
    Q = zeros([k,n],dtype='float64')
    oldloss = sys.maxint
    #初始化P和Q矩阵
    for i in range(m):
        for j in range(k):
            P[i][j] = random.uniform(0, 5)
    for i in range(k):
        for j in range(n):
            Q[i][j] = random.uniform(0, 5)

    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if(juzhen[i][j] > 0):
                    error = juzhen[i][j]
                    for h in range(k):
                        error -= P[i][h]*Q[h][j]
                    #更新P矩阵和Q矩阵
                    for h in range(k):
                        P[i][h] += alpha * (2 * error * Q[h][j] - beta * P[i][h])
                        Q[h][j] += alpha * (2 * error * P[i][h] - beta * Q[h][j])

        #就是一次迭代后loss的值，即原矩阵中非零元素与预测的对应元素的差的平方
        loss = 0
        for i in range(m):
            for j in range(n):
                if(juzhen[i][j] > 0):
                    error = 0
                    for h in range(k):
                        error += P[i][h] * Q[h][j]
                    loss += pow(juzhen[i][j] - error, 2)
                    for h in range(k):
                        loss += (beta/2) * (pow(P[i][h], 2) + pow(Q[h][j], 2))
        print "The %d epoch loss is %f" % (step, loss)
        if loss < oldloss:
            oldloss = loss
        else:
            print "已经收敛到最小,开始往大的收敛"
            break
        if(loss < 1):
            break
            '''
        if(step % 1000 == 0):
            print "loss:",loss, "\n"
        print step,","
        '''
    #保存分解之后的隐向量矩阵和重建之后的矩阵
    np.savetxt('P.csv', P, delimiter=',')
    np.savetxt('Q.csv', Q, delimiter=',')
    builted_juzhen = zeros([943, 1682], dtype='float64')  # 定义重建之后的矩阵

    for i in range(m):
        for j in range(n):
            temp = 0
            for h in range(k):
                temp += P[i][h] * Q[h][j]
            builted_juzhen[i][j] = temp
    np.savetxt('builted_juzhen.csv', builted_juzhen, delimiter=',')
    '''
    fr = open("P.csv", 'a')
    print >> fr, P
    fr.close()

    fr = open("Q.csv", 'a')
    print >> fr, Q
    fr.close()



    fr = open("builted_juzhen.csv", 'a')
    print >> fr, builted_juzhen
    fr.close()
    '''


if __name__ == '__main__':
    juzhen = dataProcess()
    matrix_fraction(juzhen, 100, 5000, 0.0002, 0.0002)
