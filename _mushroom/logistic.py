# -*- coding: utf-8 -*-
import numpy as np
from mushroom import *
import math
#数据预处理模块

#求出系数，大于0.5分一类，小于0.5分另一类
def sigmoid(inX):
    return 1.0/(1+math.exp(-inX))


def stocGradAscent1(dataMatrix, classLabels, numIter=150):#默认迭代150
    m,n = np.shape(dataMatrix)
    print(len(classLabels))
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
def colicTest(trainTime):
    #从mushrooms.py中读取文件
    trainingSet,trainingLabels,testSet,testLabels=dataset_load()
    #训练系数，可以改最后一个迭代次数查看不同效果
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, trainTime)
    testlist=[];
    #写文件
    f=open('mrtest.txt','w')
    for i in range(len(testLabels)):
        testlist.append([sigmoid(sum(np.array(testSet[i])*trainWeights)),testLabels[i]])
    #排序，按sigmord从高到低
    sortlist=sorted(testlist,key=lambda test:test[0],reverse=True)
    for i in range(len(testLabels)):
        f.write('x{a} {sig:.10f} {l}\n'.format(a=i,sig=sortlist[i][0],l=sortlist[i][1]))
    f.close()
    return
