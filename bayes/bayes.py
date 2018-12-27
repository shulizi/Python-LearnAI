# _*_coding:utf-8 _*_
import urllib2
import numpy
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
def get_mean_stan(data):
    narray = numpy.array(data)
    mean = 1.0*narray.sum()/len(data)
    stan = math.sqrt(narray.var())
    return mean,stan
def get_p_den(mean,stan,x):
    sq = pow((x-mean)/stan,2)
    ex= math.exp((-0.5)*sq)
    return 1/(math.sqrt(2*math.pi)*stan)*ex
def get_data(path):
    fp=open(path,'r')
    data_list = fp.readlines()
    truth_value = []
    data = []
    for item in data_list:
        data.append(int(item.strip().split()[0]))           
    return data


def get_xr_yr(mean,stan,data):
    xr = []
    yr = []
    for item in data:
        xr.append(item)
        yr.append(get_p_den(mean,stan,item))
    return xr,yr
def classify(x,threshold,mean1,stan1,mean2,stan2):
    pre_p = get_p_den(mean1,stan1,x)
    pre_p2 = get_p_den(mean2,stan2,x)
    if pre_p/pre_p2 > threshold :
        return 1
    else:
        return 0
def get_fpr_tpr(mean1,stan1,mean2,stan2,data,label,threshold):
    fp = 0
    tp = 0
    fptn = 0
    tpfn = 0
    for i in range(len(label)):
        forecast = classify(data[i],threshold,mean1,stan1,mean2,stan2)
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
def ROC():
    data1 =  get_data('boy.txt')
    mean1,stan1 = get_mean_stan(data1)
    data2 =  get_data('girl.txt')
    mean2,stan2 = get_mean_stan(data2)
    
    data = numpy.append(data1,data2,axis=0)
    label = numpy.append(numpy.exp(numpy.array(data1)*0),numpy.array(data2)*0)
 
    x = []
    y = []

    scores = []
    for i in range(len(data)):
        threshold = get_p_den(mean1,stan1,data[i])/get_p_den(mean2,stan2,data[i])
        scores.append(threshold)
        fpr,tpr = get_fpr_tpr(mean1,stan1,mean2,stan2,data,label,threshold)
        x.append(fpr)
        y.append(tpr)
    print scores
    auc = roc_auc_score(label,scores)
    plt.text(0.6,0.5,'AUC={:.6f}'.format(auc))
    plt.plot([0,1],[0,1],'r+--')
    plt.scatter(x,y)
    plt.title('ROC')
    plt.grid(True)
    plt.show()
    
def draw():
    plt.title('probability density')
    data1 =  get_data('boy.txt')
    mean1,stan1 = get_mean_stan(data1)
    data2 =  get_data('girl.txt')
    mean2,stan2 = get_mean_stan(data2)
    x,y = get_xr_yr(mean1,stan1,data1)  
    plt.scatter(x,y,color='red')
    x,y = get_xr_yr(mean2,stan2,data2)  
    plt.scatter(x,y,color='yellow')
    plt.show()
ROC()
draw()
