# _*_coding:utf-8 _*_
import urllib2
from numpy import *
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

def id3(data,label):
    feat1_data = data[:,0]
    feat2_data = data[:,1]
    len_data = len(data)
    label0Count = 0
    label1Count = 0
    feat1_values = {}
    feat2_values = {}
    for i in range(len_data):
        if label[i] == 0:
            label0Count += 1
        else:
            label1Count += 1
        if feat1_data[i] not in feat1_values:
            feat1_values[feat1_data[i]] = []
        feat1_values[feat1_data[i]].append(i)
        
        if feat2_data[i] not in feat2_values:
            feat2_values[feat2_data[i]] = []
        feat2_values[feat2_data[i]].append(i)
    
    
    
    p0 = 1.0*label0Count/len_data
    p1 = 1.0*label1Count/len_data
    shannon_ent = -p0*math.log(p0,2)-p1*math.log(p1,2)
    
    feat1_shannon_ent = 0
    for key in feat1_values:
        len_feat_data = len(feat1_values[key])
        label0Count = 0
        label1Count = 0
        for i in feat1_values[key]:
            if label[i] == 0:
                label0Count += 1
            else:
                label1Count += 1
        p0 = 1.0*label0Count/len_feat_data
        p1 = 1.0*label1Count/len_feat_data
        if p0 == 0 or p1 == 0:
            feat_value_shannon_ent = 0
        else:
            feat_value_shannon_ent = -p0*math.log(p0,2)-p1*math.log(p1,2)
        
        feat1_shannon_ent += 1.0*len_feat_data/len_data*feat_value_shannon_ent
    
    delta_feat1_shannon_ent = shannon_ent - feat1_shannon_ent
    
    feat2_shannon_ent = 0
    for key in feat2_values:
        len_feat_data = len(feat2_values[key])
        label0Count = 0
        label1Count = 0
        for i in feat2_values[key]:
            if label[i] == 0:
                label0Count += 1
            else:
                label1Count += 1
        p0 = 1.0*label0Count/len_feat_data
        p1 = 1.0*label1Count/len_feat_data
        if p0 == 0 or p1 == 0:
            feat_value_shannon_ent = 0
        else:
            feat_value_shannon_ent = -p0*math.log(p0,2)-p1*math.log(p1,2)
        feat2_shannon_ent += 1.0*len_feat_data/len_data*feat_value_shannon_ent
    delta_feat2_shannon_ent = shannon_ent - feat2_shannon_ent
    print "Information gain:  Discount_rate:",delta_feat1_shannon_ent,"Distance:",delta_feat2_shannon_ent
    return delta_feat1_shannon_ent,delta_feat2_shannon_ent
