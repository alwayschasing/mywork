#!/usr/bin/env python
# coding=utf-8

from neuralnet import NeuralNetwork
import tensorflow as tf
import csv

def getTrainData():
    
    with open("../data/ml-1m/rnndata2.csv","r") as fp:
        lines = csv.reader(fp)
        lines = list(lines)
