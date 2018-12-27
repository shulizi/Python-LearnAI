# _*_coding:utf-8 _*_
import urllib2
from numpy import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeavePOut

def get_data(path):
    fp=open(path,'r')
    data_list = fp.readlines()
    truth_value = []
    data = []
    for item in data_list:
        data.append(float(item.strip().split()[0]))
        data.append(float(item.strip().split()[1]))
    return array(data).reshape(len(data)/2,2)
def classify(x,pro_dir,w0,threshold):
    x = matrix(x).T
    w0 = matrix(w0).T
    if pro_dir.T*(x-w0)>log(threshold):
        return 1
    else:
        return 0
def get_fpr_tpr(data,label,pro_dir,w0,threshold):
    fp = 0
    tp = 0
    fptn = 0
    tpfn = 0
    for i in range(len(label)):
        forecast = classify(data[i],pro_dir,w0,threshold)
        if label[i] == 1:
            tpfn += 1
        else:
            fptn += 1
        if label[i] == 1 and forecast == 1:
            tp += 1
        elif label[i] == 0 and forecast == 1:
            fp += 1
    
    fpr = 1.0*fp/fptn
    tpr = 1.0*tp/tpfn
    return fpr,tpr
def get_distance(x1,x2):
    return sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
def get_min_distance_x(x0,data):
    min_distance = float(get_distance(x0,data[0]))
    min_distance_x= data[0]
    get_distance(data[0],data[0])
    for i in data:
        if(float(get_distance(x0,i))<min_distance):
            min_distance = float(get_distance(x0,i))
            min_distance_x = i
    
    return min_distance_x
    
def draw():
    
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    data = append(data1,data2,axis=0)
    
    print get_min_distance_x([177,60],data)

draw()
