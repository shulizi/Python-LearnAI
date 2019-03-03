# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
def as_num(x):
    y = '{:.10f}'.format(x)
    return (y)
def get_data(path):
    fp=open(path,'r')
    data_list = fp.readlines()
    truth_value = []
    data = []
    for item in data_list:
        data.append(1)
        data.append(float(item.strip().split()[0]))
        data.append(float(item.strip().split()[1]))
        
    
    return array(data).reshape(len(data)/3,3)
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) 
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    
    return 1.0/(1+exp(-inX))

def oneScale(data):
    data = array(data)
    for i in range(1,shape(data)[1]):
        data[:,i] = map(lambda x:(x-data[:,i].min())/(data[:,i].max()-data[:,i].min()),data[:,i])
    return data

def gradAscent(dataMatrix, labelMat):
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.01
    maxCycles = 200
    
    weights = ones(n)
    for i in range(maxCycles):
        for k in range(m):
            h = sigmoid(sum(dataMatrix[k]*weights))
            error = labelMat[k] - h
            weights = weights + alpha * error * dataMatrix[k]
    
    
    return weights

def get_x_y():
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    dataMat = append(data1,data2,axis=0)
    labelMat = append(array(exp(data1*0)[:,1]),array(data2*0)[:,2])
    #dataMat,labelMat = loadDataSet('txt_set.txt')
    dataMat = oneScale(dataMat)
    
    #weights = gradAscent(dataMat,labelMat)
    #output_roc(dataMat,labelMat,weights)
    #weights = weights.getA()
    dataMat1 = oneScale(data1)
    dataMat2 = oneScale(data2)
    mean1 = mean(dataMat1,axis = 0)
    mean2 = mean(dataMat2,axis = 0)
    
    #dataArr = array(dataMat)
    n = shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    k = 0.1
    x = []
    y = []
    for r in range(n):
        dataM = dataMat
        w = mat(eye((n)))
        point = dataM[r]
            
        for j in range(n):
            diffMat = mat(point - dataMat[j,:])
            w[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
        dataM = w*dataM
        weights = gradAscent(dataM,labelMat)
        
        
        
        x0 = dataMat[r,1]
        #x = arange(0,1,0.01)
        y0 = (-weights[0]-weights[1] * x0)/weights[2]
        if x0>0 and x0<1 and y0>0 and y0<1:
            x.append(x0)
            y.append(y0)
        
    '''
    x = arange(0,1,0.01)
    y = (-weights[0]-weights[1] * x)/weights[2]
    '''
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i,1]);ycord1.append(dataMat[i,2])
        else:
            xcord2.append(dataMat[i,1]);ycord2.append(dataMat[i,2])
    
    return xcord1,ycord1,xcord2,ycord2,x,y
def output_roc(dataMat,labelMat,weights):
    with open('roc.txt','w') as output:
        h = sigmoid(dataMat*weights)
        h = array(h)
        a = sorted(map(lambda x,y:str('{:.10f}'.format(x[0]))+" "+str(y),h,labelMat),reverse=True)
        for i in range(len(a)):
            output.write("x"+str(i)+" "+a[i]+"\n")
def draw():
    plt.title('logistic')
    x1,y1,x2,y2,x,y = get_x_y()
    plt.scatter(x1,y1,color='blue')
    plt.scatter(x2,y2,color='red')
    
    plt.plot(x,y,color='yellow')
    plt.show()
draw()
