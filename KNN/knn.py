#!/usr/bin/env python
# coding=utf-8

import numpy as np
import csv


def getTrainData():
    fp = open("trainMF","r") 
    n_user = 6041
    n_item = 3953

    R_matrix = np.zeros([n_user,n_item],dtype=np.int32)

    lines = list(csv.reader(fp))

    for line in lines:
        user = int(line[0])
        n = len(line)
        for i in range(0,n):
            item = int(line[2*i+1])
            rating = int(line[2*i+2])
            R_matrix[user][item] = rating

    fp.close()
    return R_matrix

def getUserDistance():
    pass

def getItemDistance():
    pass



    
