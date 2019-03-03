# _*_coding:utf-8 _*_
import urllib2
from numpy import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

def get_data(path):
    fp=open(path,'r')
    data_list = fp.readlines()
    truth_value = []
    data = []
    for item in data_list:
        data.append(float(item.strip().split()[0]))
        data.append(float(item.strip().split()[1]))
        data.append(float(item.strip().split()[2]))
    return array(data).reshape(len(data)/3,3)
def classify(min_distance_index,label):
    boy_n = 0
    girl_n = 0
    
    for i in range(len(min_distance_index)):
        if label[min_distance_index[i]] == 1:
            boy_n += 1
        else:
            girl_n += 1
    
    if boy_n > girl_n:
        return 1
    else:
        return 0
def classify_surface(x0,min_distance_index,data,label):
    boy_n = 0
    girl_n = 0

    if (label[min_distance_index[0]] == 1 and label[min_distance_index[1]] == 0)\
       or (label[min_distance_index[0]] == 0 and label[min_distance_index[1]] == 1):
        if get_distance(data[min_distance_index[0]],x0) - get_distance(data[min_distance_index[1]],x0)  < 0.0000000001:
            return 1
        else:
            return 0
    return 0   
def get_distance(x1,x2):
    a = 0
    for i in range(len(x1)):
        a += (x1[i]-x2[i])**2
    return sqrt(a)
def get_min_distance_index(x0,data,k):
    min_distance_index = []
    for i in range(k):
        min_distance_index.append(i)

    for i in range(len(data)):
        
        max_of_min_distance_index = 0
        max_of_min_distance = get_distance(x0, data[min_distance_index[0]])
        for j in range(k):
            min_distance_x = list(data[min_distance_index[j]])
            if(float(get_distance(x0,min_distance_x))>max_of_min_distance):
                max_of_min_distance_index = j
                max_of_min_distance = float(get_distance(x0,min_distance_x))
        
        
        if float(get_distance(x0,data[i])) < max_of_min_distance:
            min_distance_index[max_of_min_distance_index] = i
    
    return min_distance_index
def get_max_distance(x0,data,x_array):
    max_distance = get_distance(x0,data[x_array[0]])
    for j in range(len(x_array)):
        max_distance_x = list(data[x_array[j]])
        if float(get_distance(x0,max_distance_x))>max_distance:\
            max_distance = float(get_distance(x0,max_distance_x))    
    return  max_distance
def get_errorrate(classify_array,label):
    fp = 0
    fn = 0
    fptn = 0
    tpfn = 0

    
    for i in range(len(label)):
        if label[i] == 1:
            tpfn += 1
        else:
            fptn += 1
        if label[i] == 1 and classify_array[i] == 0:
            fn += 1
        elif label[i] == 0 and classify_array[i] == 1:
            fp += 1
    print classify_array,label
    errorrate = 1.0*(fn+fp)/len(label)
    return errorrate
def errorrate_line():
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    data = append(data1,data2,axis=0)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])

    x = []
    y = []
    for k in range(1,80,2):
        classify_array = []
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(data):
            data_train, data_test = data[train_index], data[test_index]
            min_distance_index = get_min_distance_index(data_test[0],data_train,k)
            classify_array.append(classify(min_distance_index,label))
            
        
        errorrate = get_errorrate(classify_array,label)
        x.append(k)
        y.append(errorrate)
    
    plt.plot(x,y,color='darkred')
    
    plt.xlabel("k")
    plt.ylabel("error rate")
    plt.grid(True)
    plt.show()

def draw():
    ax = plt.subplot(111,projection='3d')
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    data = append(data1,data2,axis=0)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])

    x = data1[:,0]
    y = data1[:,1]
    z = data1[:,2]
    ax.scatter(x,y,z,c='b')

    x = data2[:,0]
    y = data2[:,1]
    z = data2[:,2]
    ax.scatter(x,y,z,c='r')

    
    x0 = [177,50,40]
    min_distance_index = get_min_distance_index(x0,data,3)
    
    for i in range(len(min_distance_index)):
        a = data[min_distance_index[i]][0] - x0[0]
        b = data[min_distance_index[i]][1] - x0[1]
        c = data[min_distance_index[i]][2] - x0[2]
        x = linspace(data[min_distance_index[i]][0],x0[0],50)
        y = (x - x0[0])/a *b + x0[1]
        z = (x - x0[0])/a *c + x0[2]
        plt.plot(x,y,z,c='g')
    
    
    
    plt.show()
    print classify(min_distance_index,label)

def draw_classify_surface():
    ax = plt.subplot(111,projection='3d')
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    data = append(data1,data2,axis=0)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])

    x = data1[:,0]
    y = data1[:,1]
    z = data1[:,2]
    ax.scatter(x,y,z,c='b')

    x = data2[:,0]
    y = data2[:,1]
    z = data2[:,2]
    ax.scatter(x,y,z,c='r')

    
    I = linspace(160,180,20)
    J = linspace(45,65,20)
    Xl,Yl = meshgrid(I,J)
    X=[]
    Y=[]
    Z=[]
    Xl = reshape(Xl,len(Xl)*len(Xl),1)
    Yl = reshape(Yl,len(Yl)*len(Yl),1)
    for zi in range(38,45):
        for i in range(len(Xl)):
            min_distance_index = get_min_distance_index([Xl[i],Yl[i],zi],data,2)
            if classify_surface([Xl[i],Yl[i],zi],min_distance_index,data,label):
                X.append(Xl[i])
                Y.append(Yl[i])
                Z.append(zi)
        
    
    
    ax.scatter(X,Y,Z,c='y')
    '''
    n = int(sqrt(len(array(X))))
    X = X[0:n**2]
    Y = Y[0:n**2]
    Z = Z[0:n**2]
    X = array(X).reshape(n,-1)
    Y = array(Y).reshape(n,-1)
    Z = array(Z).reshape(n,-1)
    ax.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap = plt.cm.coolwarm)
    '''
    
    plt.show()
    print classify(min_distance_index,label)
#errorrate_line()
#draw()
draw_classify_surface()

