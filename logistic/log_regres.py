# -*- coding: utf-8 -*-
from numpy import *


def as_num(x):
    y = '{:.10f}'.format(x)
    return (y)
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('txt_set.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1, float(lineArr[0]), float (lineArr[1])])
        labelMat.append(int (lineArr[2]))
    fr.close()
    return dataMat,labelMat

def sigmoid(inX):
    m,n = shape(inX)
    arr = []
    for row in range(m):
        row_arr = []
        for col in range(n):           
            if inX[row,col] > 0 :
                a = 1.0/(1+exp(-inX[row,col]))

            else:
                a = exp(1.0*inX[row,col])/(1+exp(inX[row,col]))

            row_arr.append(a)
        arr.append(row_arr)
    
    return mat(arr)
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights

def get_x_y():
    dataMat,labelMat = loadDataSet()
    weights = gradAscent(dataMat,labelMat)
    output_roc(dataMat,labelMat,weights)
    weights = weights.getA()
    
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    x = arange(150,180,0.1)
    y = (-weights[0]-weights[1] * x)/weights[2]
    return xcord1,ycord1,xcord2,ycord2,x,y
def output_roc(dataMat,labelMat,weights):
    with open('roc.txt','w') as output:
        h = sigmoid(dataMat*weights)
        h = array(h)
        a = sorted(map(lambda x,y:str('{:.10f}'.format(x[0]))+" "+str(y),h,labelMat),reverse=True)
        for i in range(len(a)):
            output.write("x"+str(i)+" "+a[i]+"\n")
