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
def get_errorrate(data,label,pro_dir,w0,threshold):
    fp = 0
    fn = 0
    fptn = 0
    tpfn = 0

    
    for i in range(len(label)):
        forecast = classify(data[i],pro_dir,w0,threshold)
        if label[i] == 1:
            tpfn += 1
        else:
            fptn += 1
        if label[i] == 1 and forecast == 0:
            fn += 1
        elif label[i] == 0 and forecast == 1:
            fp += 1
    fpr = 1.0*fp/fptn
    fnr = 1.0*fn/tpfn
    fprfnr = [fpr,fnr]
    
    return fprfnr
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

    k = pro_dir[1]/pro_dir[0]
    k = array(k)[0]
    p_x,p_y = get_projection_xy(data[:,0],data[:,1],k)
    
    x = []
    y = []

    scores = []

    w0 = [mean(p_x),mean(p_y)]
    
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
def errorrate_line():
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    mean1,scatter1 = get_mean_scatter(data1)
    mean2,scatter2 = get_mean_scatter(data2)
    data = append(data1,data2,axis=0)
    scatter = scatter1 + scatter2
    pro_dir = get_projection_direction(scatter,mean1,mean2)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])

    k = pro_dir[1]/pro_dir[0]
    k = array(k)[0]
    p_x,p_y = get_projection_xy(data[:,0],data[:,1],k)
    
    xy = []


    w0 = [mean(p_x),mean(p_y)]
    
    for i in range(len(data)):
        threshold = get_threshold(data[i],pro_dir,w0)
        errorrate = get_errorrate(data,label,pro_dir,w0,threshold)
        xy.append(errorrate)
        
    xy = array(xy)
    xy = xy[lexsort(xy.T)]
    x = array(xy[:,0])
    y = array(xy[:,1])
    plt.plot([0,1],[0,1],'r+--')
    plt.plot(x,y,color='darkred')
    
    
    plt.title('error rate')
    plt.grid(True)
    plt.show()

def get_projection_xy(x,y,k):
    p_x = (x+k*y)/(k*k+1)
    p_y = k*p_x
    return p_x,p_y
def draw():
    
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    mean1,scatter1 = get_mean_scatter(data1)
    mean2,scatter2 = get_mean_scatter(data2)
    data = append(data1,data2,axis=0)
    scatter = scatter1 + scatter2
    pro_dir = get_projection_direction(scatter,mean1,mean2)
    k = pro_dir[1]/pro_dir[0]
    k = array(k)[0]
    x = arange(140,200,1)
    
    y = k*x
    plt.plot(x,y)#projection line
    
    
    
    x = data1[:,0]
    y = data1[:,1]
    plt.scatter(x,y,color='yellow')#boy
    
    p_x,p_y = get_projection_xy(x,y,k)
    for i in range(len(p_x)):
        p_x_x = [p_x[i],x[i]]
        p_x_y = [p_y[i],y[i]]
        plt.plot(p_x_x,p_x_y,color='orange')#vertical line to projection
    
    x2 = data2[:,0]
    y2 = data2[:,1]
    plt.scatter(x2,y2,color='blue')#girl

    p_x2,p_y2 = get_projection_xy(x2,y2,k)
    
    for i in range(len(p_x2)):
        p_x_x2 = [p_x2[i],x2[i]]
        p_x_y2 = [p_y2[i],y2[i]]
        plt.plot(p_x_x2,p_x_y2,color='darkblue')#vertical line to projection
    mean_p_x = mean(p_x)
    mean_p_y = mean(p_y)
    mean_p_x2 = mean(p_x2)
    mean_p_y2 = mean(p_y2)
    wx0 = 0.5*(mean_p_x + mean_p_x2)
    wy0 = 0.5*(mean_p_y + mean_p_y2)
    x = arange(160,180,1)
    y = -1/k*(x-wx0)+wy0
    plt.plot(x,y,color='red')#division line

    w0 = [wx0,wy0]
    plt.show()
draw()
ROC()
errorrate_line()
