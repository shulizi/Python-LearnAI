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
def get_low_dim_data(data):
    meanval = mean(data,axis=0)
    data = data - meanval
    cov_mat = cov(data.T)
    igen_vals, eigen_vecs = linalg.eig(cov_mat)
    index = argsort(igen_vals)
    n_index=index[-1:-3:-1]
    n_featVec=eigen_vecs[:, n_index]
    low_data =  dot(data,n_featVec)
    print cov(low_data[:,0]),cov(low_data[:,1])
    high_data = dot(low_data,n_featVec.T)+meanval
    return low_data,high_data

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
def get_threshold(x,pro_dir,w0):
    x = matrix(x).T
    w0 = matrix(w0).T
    return float(exp(pro_dir.T*(x-w0)))
def fisher_classify(x,pro_dir,w0,threshold):
    x = matrix(x).T
    w0 = matrix(w0).T
    if pro_dir.T*(x-w0)>log(threshold):
        return 1
    else:
        return 0
    
def get_projection_xyz(a,line):
    
    h = a*line/linalg.norm(line)
    l=linalg.norm(a,axis=1)
    h =  array(h.T)[0]
    
    a_line_l = (l**2-h.T[0]**2)**0.5
    a_line = line/linalg.norm(line)*a_line_l
    
    return a_line

def get_fisher_fpr_tpr(data,label,pro_dir,w0,threshold):
    fp = 0
    tp = 0
    fptn = 0
    tpfn = 0
    for i in range(len(label)):
        forecast = fisher_classify(data[i],pro_dir,w0,threshold)
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

def get_mean_scatter(data):
    narray = data.T
    mea = mean(narray,axis = 1)
    scatter = cov(narray)
    return matrix(mea).T,scatter

def get_projection_direction(scatter,mean1,mean2):
    return linalg.inv(scatter)*(mean1 - mean2)

def get_projection_xy(x,y,k):
    p_x = (x+k*y)/(k*k+1)
    p_y = k*p_x
    return p_x,p_y
def fisher_classify(x,pro_dir,w0,threshold):
    x = matrix(x).T
    w0 = matrix(w0).T
    if pro_dir.T*(x-w0)>log(threshold):
        return 1
    else:
        return 0
def get_threshold(x,pro_dir,w0):
    x = matrix(x).T
    w0 = matrix(w0).T
    return float(exp(pro_dir.T*(x-w0)))
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
def classify_surface(x0,min_distance_index,data,label):
    boy_n = 0
    girl_n = 0
    
    if (label[min_distance_index[0]] == 1 and label[min_distance_index[1]] == 0)\
       or (label[min_distance_index[0]] == 0 and label[min_distance_index[1]] == 1):
        if abs(get_distance(data[min_distance_index[0]],x0) - get_distance(data[min_distance_index[1]],x0) ) < 0.01:
            
            return 1
        else:
            return 0
    return 0
def ROC():
    data1 =  get_data('boy.txt')
    mean1,cov1 = get_mean_cov(data1)
    data2 =  get_data('girl.txt')
    mean2,cov2 = get_mean_cov(data2)

    
    data = append(data1,data2,axis=0)
    low_data,high_data = get_low_dim_data(data)
    
    
    data_test1 =  get_data('boy_test.txt')
    data_test2 =  get_data('girl_test.txt')
    data_test = append(data_test1,data_test2,axis=0)
    label = append(array(exp(data_test1*0)[:,0]),array(data_test2*0)[:,0])
    
    low_data_test,high_data_test = get_low_dim_data(data_test)
    
    x = []
    y = []

    scores = []
    for i in range(len(data_test)):
        threshold = get_p_den(mean1,cov1,data_test[i])/get_p_den(mean2,cov2,data_test[i])
        scores.append(int(threshold))
        fpr,tpr = get_fpr_tpr(mean1,cov1,mean2,cov2,data_test,label,threshold)
        x.append(fpr)
        y.append(tpr)
    plt.scatter(x,y)
    auc = roc_auc_score(label,scores)
    plt.text(0.1,0.5,'AUC={:.6f}'.format(auc),color='blue')

    
    low_data1 = low_data[:len(data1)]
    low_data2 = low_data[len(data1):len(data1)+len(data2)]
    mean1,cov1 = get_mean_cov(low_data1)
    mean2,cov2 = get_mean_cov(low_data2)
    x = []
    y = []
    scores = []
    for i in range(len(low_data_test)):
        threshold = get_p_den(mean1,cov1,low_data_test[i])/get_p_den(mean2,cov2,low_data_test[i])
        scores.append(int(threshold))
        fpr,tpr = get_fpr_tpr(mean1,cov1,mean2,cov2,low_data_test,label,threshold)
        x.append(fpr)
        y.append(tpr)
    plt.scatter(x,y,color='red',marker = '+')
    auc = roc_auc_score(label,scores)
    
    
    plt.text(0.1,0.4,'AUC={:.6f}'.format(auc),color='red')
    plt.plot([0,1],[1,0],'y+--')
    




    mean1,scatter1 = get_mean_scatter(data1)
    mean2,scatter2 = get_mean_scatter(data2)
    scatter = scatter1 + scatter2
    pro_dir = get_projection_direction(scatter,mean1,mean2)
       
    p_xyz = get_projection_xyz(data,pro_dir)

    p_x = p_xyz[0]
    p_y = p_xyz[1]
    p_z = p_xyz[2]
    
    x = []
    y = []

    scores = []
    
    w0 = [mean(p_x),mean(p_y),mean(p_z)]
    
    for i in range(len(data_test)):
        threshold = get_threshold(data_test[i],pro_dir,w0)
        scores.append(float(threshold))
        fpr,tpr = get_fisher_fpr_tpr(data_test,label,pro_dir,w0,threshold)
        x.append(fpr)
        y.append(tpr)
    auc = roc_auc_score(label,scores)
    plt.text(0.1,0.3,'AUC={:.6f}'.format(auc),color='orange')
    plt.scatter(x,y,marker = '.')



    low_data1 = low_data[:len(data1)]
    low_data2 = low_data[len(data1):len(data1)+len(data2)]
    mean1,scatter1 = get_mean_scatter(low_data1)
    mean2,scatter2 = get_mean_scatter(low_data2)

    low_data_test,high_data_test = get_low_dim_data(data_test)
    
    scatter = scatter1 + scatter2
    pro_dir = get_projection_direction(scatter,mean1,mean2)
       
    p_xyz = get_projection_xyz(low_data,pro_dir)
    
    p_x = p_xyz[0]
    p_y = p_xyz[1]

    
    x = []
    y = []

    scores = []
    
    w0 = [mean(p_x),mean(p_y)]
    
    for i in range(len(low_data_test)):
        threshold = get_threshold(low_data_test[i],pro_dir,w0)
        scores.append(float(threshold))
        fpr,tpr = get_fisher_fpr_tpr(low_data_test,label,pro_dir,w0,threshold)
        x.append(fpr)
        y.append(tpr)
    auc = roc_auc_score(label,scores)
    plt.text(0.1,0.2,'AUC={:.6f}'.format(auc),color='green')
    plt.scatter(x,y,marker = 'v')



    plt.legend(['','bayes','bayes_kl' ,'fisher', 'fisher_kl'])  
    plt.title('ROC')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.grid(True)
    plt.show()

def draw():
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')
    
    data = append(data1,data2,axis=0)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])


    
    low_data,high_data = get_low_dim_data(data)
    
    x = low_data[:len(data1),0]
    y = low_data[:len(data1),1]
    plt.scatter(x,y,color='blue')

    x = low_data[len(data1):len(data1)+len(data2),0]
    y = low_data[len(data1):len(data1)+len(data2),1]
    plt.scatter(x,y,color='red')

    
    '''
    test_data1 =  get_data('boy_test.txt')
    test_data2 =  get_data('girl_test.txt')
    test_data = append(test_data1,test_data2,axis=0)
    low_data_test,high_data_test = get_low_dim_data(test_data)
    '''
    I = linspace(-15,15,100)
    J = linspace(-15,15,100)
    Xl,Yl = meshgrid(I,J)
    Xl = reshape(Xl,len(Xl)*len(Xl),1)
    Yl = reshape(Yl,len(Yl)*len(Yl),1)
    XY = []
    for i in range(len(Xl)):
        min_distance_index = get_min_distance_index([Xl[i],Yl[i]],low_data,2)
    
        if classify_surface([Xl[i],Yl[i]],min_distance_index,low_data,label):
            XY.append([Xl[i],Yl[i]])
            
    XY = array(XY)
    XY = XY[lexsort(XY.T)]
    X = array(XY[:,0])
    Y = array(XY[:,1])
    plt.plot(X,Y)
    plt.show()

    
def draw_3d():
    ax = plt.subplot(111,projection='3d')
    data1 =  get_data('boy.txt')
    data2 =  get_data('girl.txt')

    print data1,data2
    data = append(data1,data2,axis=0)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])

    x = data1[:,0]
    y = data1[:,1]
    z = data1[:,2]
    ax.scatter(x,y,z,c='b',alpha=0.2)
    x = data2[:,0]
    y = data2[:,1]
    z = data2[:,2]
    ax.scatter(x,y,z,c='r',alpha=0.2)

    low_data,high_data = get_low_dim_data(data)
    x = high_data[:len(data1),0]
    y = high_data[:len(data1),1]
    z = high_data[:len(data1),2]
    ax.scatter(x,y,z,color='green')

    x = high_data[len(data1):len(data1)+len(data2),0]
    y = high_data[len(data1):len(data1)+len(data2),1]
    z = high_data[len(data1):len(data1)+len(data2),2]
    ax.scatter(x,y,z,color='black')
    '''
    for i in range(len(data)):
        x = [data[i][0],high_data[i][0]]
        y = [data[i][1],high_data[i][1]]
        z = [data[i][2],high_data[i][2]]

        
        plt.plot(x,y,z)
        
    '''
    plt.show()

#ROC()
#draw_3d()
draw()

