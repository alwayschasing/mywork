#!/usr/bin/env python
# coding=utf-8
import csv

def getRNNtr_data():
    fp = open("/home/lrh/graduation_project/data/CiaoDVD/finUserBasedTrain.csv","r")
    sa = open("/home/lrh/graduation_project/data/CiaoDVD/rnndata1.csv","w")
    writer = csv.writer(sa)
    lines = fp.readlines()
    timestepsize = 10
    count = 0 #rnn训练数据行数
    u_n = 0 #序列长度大于10的用户数
    for line in lines:
        line = line.strip().split(',')
        k = len(line)
        # +1 表示去掉开头的用户编号
        if k < timestepsize+1: continue
        u_n += 1
        i = 1
        while i+timestepsize <= k:
            tmp = [line[0]]
            tmp.extend(line[i:i+timestepsize])
            writer.writerow(tmp)
            count += 1
            i += 1 #生成数据滑动窗口滑动步长为2
        tmp = [line[0]]
        tmp.extend(line[-timestepsize:])
        writer.writerow(tmp)
    print count
    print u_n
    fp.close()
    sa.close()

def getRNNtr_dataAll():
    fp = open("./userbased.train.csv","r")
    sa = open("./rnndata1.csv","w")
    writer = csv.writer(sa)
    lines = fp.readlines()
    timestepsize = 10
    count = 0 #rnn训练数据行数
    u_n = 0 #序列长度大于10的用户数
    for line in lines:
        line = line.strip().split(',')
        k = len(line)
        # +1 表示去掉开头的用户编号
        if k < timestepsize+1: continue
        u_n += 1
        i = 1
        while i+timestepsize <= k:
            tmp = [line[0]]
            tmp.extend(line[i:i+timestepsize])
            writer.writerow(tmp)
            count += 1
            i += 1 #生成数据滑动窗口滑动步长为2
        tmp = [line[0]]
        tmp.extend(line[-timestepsize:])
        writer.writerow(tmp)
    print count
    print u_n
    fp.close()
    sa.close()
def getRNNte_data():
    fp1 = open("./userbased.train.csv","r")
    fp2 = open("./userbased.test.csv","r")
    fpwrite = open("./rnntestdata.csv","w")
    
    content1 = fp1.readlines()
    content2 = fp2.readlines()
    writer = csv.writer(fpwrite)

    if(len(content1) != len(content2)):
        print "data got error"
        return
    nlen = len(content1)
    for i in range(nlen):
        line1 = content1[i].strip().split(',')
        line2 = content2[i].strip().split(',')
        if line1[0] != line2[0]:
            print "data got error"
            return
        if len(line1) < 10:continue

        u = line1[0]
        tmp = [u]+line1[-9:]+line2[1:]
        writer.writerow(tmp)
    fp1.close()
    fp2.close()
    fpwrite.close()

def getPaperTrainData():
    fp = open("./userbased.trainMF.csv","r")
    reader = csv.reader(fp)
    sp = open("./paperTraindata.csv","w")
    writer = csv.writer(sp)
    lines = list(reader)
    seqlen = 10
    for line in lines:
        l = len(line) #一行的长度
        #不够生成一个序列的话跳过
        if l < 2*seqlen+1: 
            print "get error data"
            return
        i = 1
        while i+2*seqlen <= l:
            tmp = [line[0]]
            tmp.extend(line[i:i+2*seqlen])
            writer.writerow(tmp)
            i += 4 #这里滑动步长仍然为1
        tmp = [line[0]]
        tmp.extend(line[-2*seqlen:])
        writer.writerow(tmp)
    fp.close()
    sp.close()

def getPaperTestData():
    """
    测试数据的输入部分包含评分,而预测部分没有评分,
    只有电影编号
    """
    fp1 = open("./userbased.trainMF.csv","r")
    fp2 = open("./userbased.test.csv","r")
    fpwrite = open("./paperTestdata.csv","w")
    
    content1 = fp1.readlines()
    content2 = fp2.readlines()
    writer = csv.writer(fpwrite)

    if(len(content1) != len(content2)):
        print "data got error"
        return
    nlen = len(content1)
    for i in range(nlen):
        line1 = content1[i].strip().split(',')
        line2 = content2[i].strip().split(',')
        if line1[0] != line2[0]:
            print "data got error"
            return
        if len(line1) < 10*2: continue

        u = line1[0]
        tmp = [u]+line1[-9*2:]+line2[1:]
        writer.writerow(tmp)
    fp1.close()
    fp2.close()
    fpwrite.close()

        
    

if __name__ == "__main__":
    #getRNNtr_data()
    #getRNNte_data()
    getPaperTrainData()
    #getPaperTestData()


