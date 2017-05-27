#!/usr/bin/env python
# coding=utf-8

import csv


def getUserBasedData():
    
    user_set = dict()
    origin = open("/home/lrh/graduation_project/data/CiaoDVD/original/movie-ratings.txt")
    reader = csv.reader(origin)
    for line in reader:
        u = line[0]
        i = line[1]
        if user_set.has_key(u):
            user_set[u].append(i)
        else:
            user_set[u] = list([u,i]);

    save = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedData.csv","w")
    writer = csv.writer(save)
    for k,v in user_set.items():
        writer.writerow(v)
    
    




def splitDataSet():
    origin = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedData.csv","r");
    reader = csv.reader(origin)

    tr_data = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedTrain.csv","w")
    te_data = open("/home/lrh/graduation_project/data/CiaoDVD/userBasedTest.csv","w")

    writer_tr = csv.writer(tr_data)
    writer_te = csv.writer(te_data)
    for line in reader:
        if len(line) >= 21:
            writer_tr.writerow(line[:-10])
            writer_te.writerow(line[-10:])


if __name__ == "__main__":
    getUserBasedData()
    splitDataSet()





