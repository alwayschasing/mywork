#!/usr/bin/env python
# coding=utf-8
fp = open("./userbased.train.csv","r")
resp = open("./itembased.train.csv","w")
items = dict()
nrow = 0
lines = fp.readlines()
for line in lines:
    line = line.strip().split(',') 
    u = line[0]
    size = len(line)
    for i in range(1,size):
        if line[i] not in items:
            items[line[i]] = [u]
        else:
            items[line[i]].append(u)

i_num = 0 #统计item的数量
for i in items:
    resp.write(i)
    i_num += 1
    for j in items[i]:
        resp.write(","+j)
    resp.write("\n")
print i_num
fp.close()
resp.close()

