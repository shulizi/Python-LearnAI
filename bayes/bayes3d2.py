# _*_coding:utf-8 _*_
import urllib2
from numpy import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
def get_mean_cov(data):
    
    narray = data.T
    mea = mean(narray,axis = 1)
    co = cov(narray)/len(narray)
    for i in range(len(co)):
        for j in range(len(co)):
            if i!=j :
                co[i][j] = 0

    return mea,co
def get_p_den(mean,cov,x):
    
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
def classify(x,threshold,mean1,cov1,mean2,cov2):
    gx1 = get_p_den(mean1,cov1,x)
    gx2 = get_p_den(mean2,cov2,x)
    
    if gx1 / gx2 > threshold:
        return 1
    else:
        return 0
def get_fpr_tpr(mean1,cov1,mean2,cov2,data,label,threshold):
    fp = 0
    tp = 0
    fptn = 0
    tpfn = 0
    for i in range(len(label)):
        forecast = classify(data[i],threshold,mean1,cov1,mean2,cov2)
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
    mean1,cov1 = get_mean_cov(data1)
    data2 =  get_data('girl.txt')
    mean2,cov2 = get_mean_cov(data2)
    data = append(data1,data2,axis=0)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])
    x = []
    y = []

    scores = []
    for i in range(len(data)):
        threshold = get_p_den(mean1,cov1,data[i])/get_p_den(mean2,cov2,data[i])
        scores.append(int(threshold))
        fpr,tpr = get_fpr_tpr(mean1,cov1,mean2,cov2,data,label,threshold)
        x.append(fpr)
        y.append(tpr)

    auc = roc_auc_score(label,scores)
    plt.text(0.6,0.5,'AUC={:.6f}'.format(auc))
    plt.plot([0,1],[0,1],'r+--')
    plt.scatter(x,y)
    plt.title('ROC')
    plt.grid(True)
    plt.show()
    
def draw():
    
    ax = plt.subplot(111,projection='3d')
    data1 =  get_data('boy.txt')
    mean1,cov1 = get_mean_cov(data1)
    data2 =  get_data('girl.txt')
    mean2,cov2 = get_mean_cov(data2)

    x = data1[:,0]
    y = data1[:,1]
    z=[]
    for i in range(len(x)):
        z.append(get_p_den(mean1,cov1,[x[i],y[i]]))
    ax.scatter(x,y,z,c='r')
    x = data2[:,0]
    y = data2[:,1]
    z=[]
    for i in range(len(x)):
        z.append(get_p_den(mean2,cov2,[x[i],y[i]]))
    ax.scatter(x,y,z,c='g')
            
    
    I = linspace(140,200,50)
    J = linspace(30,90,50)
    X,Y = meshgrid(I,J)
    M = []
    N = []
    Z = []
    Z = []
    for i in range(len(I)):
        for j in range(len(J)):
            if get_p_den(mean1,cov1,[X[i][j],Y[i][j]])>get_p_den(mean2,cov2,[X[i][j],Y[i][j]]):
                Z.append(get_p_den(mean1,cov1,[X[i][j],Y[i][j]]))
            else :
                Z.append(get_p_den(mean2,cov2,[X[i][j],Y[i][j]]))
            if (1<get_p_den(mean1,cov1,[X[i][j],Y[i][j]]) / get_p_den(mean2,cov2,[X[i][j],Y[i][j]])<1.2):
                M.append(X[i][j])
                N.append(Y[i][j])
 
    ax.plot(M,N,'b')
    Z = reshape(Z,(len(I),len(J)))
    ax.plot_surface(array(X),array(Y),Z,rstride = 1, cstride = 1,cmap = plt.cm.coolwarm)
    ax.contour(X,Y,Z,offset=0,cmap='coolwarm')
 
    plt.show()

ROC()
#draw()

