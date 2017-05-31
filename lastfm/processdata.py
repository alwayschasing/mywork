#!/usr/bin/env python
# coding=utf-8

import csv
import time


def getUserBasedData():
    
    """
    每个用户对应一个字典,字典又对应一个字典,存的item,time对
    """
    user_set = dict()
    origin = open("/home/lrh/graduation_project/data/")
    reader = csv.reader(origin)
    for line in reader:
        u = line[0]
        i = line[1]
        t = line[-1]
        r = line[-2]
        t = int(time.mktime(time.strptime(t,"%Y-%m-%d")))

        if user_set.has_key(u):
            user_set[u][i] = (r,t)
        else:
            user_set[u] = {i:(r,t)};

    #需要根据时间做一个排序
    save = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedData.csv","w")
    writer = csv.writer(save)

    for k,v in user_set.items():
        if len(v) >= 20: 
            tmp = v.items()
            tmp = sorted(tmp,key=lambda a:a[1][1])
            seq = [k]

            for _i in tmp:
                #电影与评分
                seq.append(_i[0])
                seq.append(_i[1][0])

            writer.writerow(seq)

    origin.close()
    save.close()
            
    
    
def splitDataSet():
    origin = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedData.csv","r");
    reader = csv.reader(origin)

    tr_data = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedTrain.csv","w")
    te_data = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedTest.csv","w")

    mf_tr = open("/home/lrh/graduation_project/data/CiaoDVD/mf_tr.txt","w")
    mf_te = open("/home/lrh/graduation_project/data/CiaoDVD/mf_te.txt","w")

    writer_tr = csv.writer(tr_data)
    writer_te = csv.writer(te_data)
    for line in reader:
        if len(line) >= 41:
            n = len(line)/2
            u = line[0]
            tr = [u]
            te = [u]
            for i in range(n-10):
                tr.append(line[i*2+1])
                mf_tr.write("%s %s %s\n"%(u,line[i*2+1],line[i*2+2]))
            for i in range(n-10,n):
                te.append(line[i*2+1])
                mf_te.write("%s %s %s\n"%(u,line[i*2+1],line[i*2+2]))
            writer_tr.writerow(tr)
            writer_te.writerow(te)

    origin.close()
    tr_data.close()
    te_data.close()
    mf_tr.close()
    mf_te.close()

#对用户与电影编码
def codeUserItem():
    origin = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedData.csv","r")
    users = dict()
    items = dict()
    u = 1
    i = 1
    reader = csv.reader(origin)
    for line in reader:
        if not users.has_key(line[0]):
            users[line[0]] = u
            u += 1
        n = len(line)/2

        for k in range(n):
            if not items.has_key(line[k*2+1]):
                items[line[k*2+1]] = i
                i += 1
    print "max u,i is %d,%d"%(u-1,i-1)
    origin.close()
    return users,items

def getfinalData():

    users,items = codeUserItem()
    tr_data = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedTrain.csv","r")
    te_data = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedTest.csv","r")


    readertr = csv.reader(tr_data)
    readerte = csv.reader(te_data)


    savetr = open("/home/lrh/graduation_project/data/CiaoDVD/finUserBasedTrain.csv","w")
    savete = open("/home/lrh/graduation_project/data/CiaoDVD/finUserBasedTest.csv","w")


    writertr = csv.writer(savetr)
    writerte = csv.writer(savete)

    for line in readertr:
        u = users[line[0]]
        tmp = [items[i] for i in line[1:]]
        tmp = [u] + tmp
        writertr.writerow(tmp)

    for line in readerte:
        u = users[line[0]]
        tmp = [items[i] for i in line[1:]]
        tmp = [u] + tmp
        writerte.writerow(tmp)

    tr_mf = open("/home/lrh/graduation_project/data/CiaoDVD/mf_tr.txt","r")
    te_mf = open("/home/lrh/graduation_project/data/CiaoDVD/mf_te.txt","r")

    save_tr_mf = open("/home/lrh/graduation_project/data/CiaoDVD/finMFtr.txt","w")
    save_te_mf = open("/home/lrh/graduation_project/data/CiaoDVD/finMFte.txt","w")

    trlines = tr_mf.readlines()
    for line in trlines:
        line = line.strip().split()
        save_tr_mf.write("%s %s %s\n"%(users[line[0]],items[line[1]],line[2]))
    
    telines = te_mf.readlines()
    for line in telines:

        line = line.strip().split()
        save_te_mf.write("%s %s %s\n"%(users[line[0]],items[line[1]],line[2]))


if __name__ == "__main__":
    #getUserBasedData()
    #splitDataSet()
    codeUserItem()
    #getfinalData()





