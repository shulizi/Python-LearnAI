# _*_coding:utf-8 _*_
import urllib2
from numpy import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeavePOut
def get_mean_scatter(data):
    
    narray = data.T
    mea = mean(narray,axis = 1)
    scatter = cov(narray)

    return matrix(mea).T,scatter

def get_projection_direction(scatter,mean1,mean2):
    return linalg.inv(scatter)*(mean1 - mean2)
def get_posterior_probability(mean,cov,x):
    x = matrix(x)
    ex =exp(-(x-mean)*linalg.inv(cov)*(x-mean).T/2)
    return 1/(2*math.pi*(linalg.det(cov)**0.5))*ex
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
def get_threshold(x,pro_dir,w0):
    x = matrix(x).T
    w0 = matrix(w0).T
    return float(exp(pro_dir.T*(x-w0)))
def ROC():
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    mean1,scatter1 = get_mean_scatter(data1)
    mean2,scatter2 = get_mean_scatter(data2)
    data = append(data1,data2,axis=0)
    scatter = scatter1 + scatter2
    pro_dir = get_projection_direction(scatter,mean1,mean2)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])

    p_x,p_y,p_z = get_projection_xyz(data,pro_dir)
    
    
    x = []
    y = []

    scores = []
    
    w0 = [mean(p_x),mean(p_y),mean(p_z)]
    
    for i in range(len(data)):
        threshold = get_threshold(data[i],pro_dir,w0)
        scores.append(float(threshold))
        fpr,tpr = get_fpr_tpr(data,label,pro_dir,w0,threshold)
        x.append(fpr)
        y.append(tpr)
    auc = roc_auc_score(label,scores)
    plt.text(0.6,0.5,'AUC={:.6f}'.format(auc))
    plt.plot([0,1],[0,1],'r+--')
    plt.scatter(x,y)
    plt.title('ROC')
    plt.grid(True)
    plt.show()

def get_projection_xyz(a,line):
    
    h = a*line/linalg.norm(line)
    l=linalg.norm(a,axis=1)
    h =  array(h.T)[0]
    
    a_line_l = (l**2-h.T[0]**2)**0.5
    a_line = line/linalg.norm(line)*a_line_l
    
    return a_line[0],a_line[1],a_line[2]

#draw()
ROC()
errorrate_line()
