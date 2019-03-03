#!/usr/bin/python
# -*- coding: utf-8 -*
import re
import math
import datetime as dt
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from pandas import  DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer#处理缺失数据 
from sklearn.metrics import roc_curve, auc

def get_date_gap(date_date_received):
    date,date_received=date_date_received.split('-')
    date_gap = dt.date(int(date[0:4]),int(date[4:6]),int(date[6:8]))-\
               dt.date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))
    date_gap = date_gap.days
    return date_gap
def is_weekend(date_received):
    date_received = dt.date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8])).weekday()+1
    if date_received == 6 or date_received == 7:
        return 1
    else:
        return 0
    
def dataset_load():
    patt = r'.*\:.*'
    datas=pd.read_csv(r'ccf_offline_stage1_train.csv')
    datas=datas.loc[datas['Coupon_id'].notnull()]
    X=datas.head(2000)
    '''
    X=datas.drop('Date',axis=1)
    X=X.drop('Date_received',axis=1)
    X=X.drop('User_id',axis=1)
    X=X.drop('Merchant_id',axis=1)
    X=X.drop('Coupon_id',axis=1)
    '''

    #Coupons
    #优惠券优惠率
    #是否是周末发的优惠券
    #满减优惠券满的值
    #满减优惠券减的值
    #是否是打折的优惠券
    x0=X[['Coupon_id']].copy()
    x0['received_count'] = 1
    x0 = x0.groupby('Coupon_id').agg('sum').reset_index()

    
    x1=X.copy().reset_index()
    x1['discount_rate'] = map(lambda x:1-float(x.split(':')[1])/float(x.split(':')[0]) \
                              if re.match(patt,x) else float(x),x1['Discount_rate'])
    x1['is_weekend']=x1['Date_received'].astype('str')
    x1.is_weekend = x1.is_weekend.apply(is_weekend)
    
    x1['full'] = map(lambda x:float(x.split(':')[0])\
                     if re.match(patt,x) else None,x1['Discount_rate'])
    x1['discount'] = map(lambda x:float(x.split(':')[1])\
                     if re.match(patt,x) else None,x1['Discount_rate'])
    x1['is_percent'] = map(lambda x:0 if re.match(patt,x) else 1,x1['Discount_rate'])
    
    x1 = x1[['Coupon_id','discount_rate','is_weekend','full','discount','is_percent']]
    x1 = x1.groupby('Coupon_id').agg('mean').reset_index()
    
    
    #同种优惠券被使用次数/被领取次数
    x2 = X[['Coupon_id','Date']].copy()
    x2 = x2.loc[x2['Date'].notnull()]
    x2['use_count'] = 1
    x2 = x2.groupby('Coupon_id')['use_count'].agg('sum').reset_index()
    x2 = pd.merge(x0,x2,on='Coupon_id',how='left')
    x2['use_rate'] = x2.use_count/x2.received_count
    x2 = x2[['Coupon_id','use_rate']]

    #领取后15日内同种优惠券被使用次数/被领取次数
    x3 = X[['Coupon_id','Date','Date_received']].copy()
    x3 = x3.loc[x3['Date'].notnull()]
    x3.Date = x3.Date.astype('str')
    x3.Date_received = x3.Date_received.astype('str')
    x3['Date_gap'] = x3.Date+'-'+x3.Date_received
    x3.Date_gap = x3.Date_gap.apply(get_date_gap)
    x3.Date_gap = x3.Date_gap.apply(lambda x:str(x) if x<=15 else '')
    x3 = x3.groupby('Coupon_id').agg({'Date_gap':lambda x:len(filter(None,x))}).reset_index()
    x3.rename(columns={'Date_gap':'use_15day_count'},inplace=True)
    x3 = pd.merge(x0,x3,on='Coupon_id',how='left')
    x3['use_15day_rate'] = x3.use_15day_count/x3.received_count
    x3 = x3[['Coupon_id','use_15day_rate']]

    coupon_feature = pd.merge(x1,x2,on='Coupon_id',how='left')
    coupon_feature = pd.merge(coupon_feature,x3,on='Coupon_id',how='left')
    
    
    
    #User
    #用户使用优惠券数/用户领取优惠券数
    x0=X[['User_id']].copy()
    x0['received_count'] = 1
    x0 = x0.groupby('User_id').agg('sum').reset_index()

    x1 = X[['User_id','Date']].copy()
    x1 = x1.loc[x1['Date'].notnull()]
    x1['use_count'] = 1
    x1 = x1.groupby('User_id')['use_count'].agg('sum').reset_index()
    x1 = pd.merge(x0,x1,on='User_id',how='left')
    x1['user_coupons_use_rate'] = x1.use_count/x1.received_count
    x1 = x1[['User_id','user_coupons_use_rate']]
    

    #15日内用户使用优惠券数/用户领取优惠券数
    x2 = X[['User_id','Date','Date_received']].copy()
    x2 = x2.loc[x2['Date'].notnull()]
    x2.Date = x2.Date.astype('str')
    x2.Date_received = x2.Date_received.astype('str')
    x2['Date_gap'] = x2.Date+'-'+x2.Date_received
    x2.Date_gap = x2.Date_gap.apply(get_date_gap)
    x2.Date_gap = x2.Date_gap.apply(lambda x:str(x) if x<=15 else '')
    x2 = x2.groupby('User_id').agg({'Date_gap':lambda x:len(filter(None,x))}).reset_index()
    x2.rename(columns={'Date_gap':'user_15day_use_count'},inplace=True)
    x2 = pd.merge(x0,x2,on='User_id',how='left')
    x2['user_15day_coupns_use_rate'] = x2.user_15day_use_count/x2.received_count
    x2 = x2[['User_id','user_15day_coupns_use_rate']]

    #用户使用优惠券平均优惠幅度
    #用户使用优惠券最小优惠幅度(max)
    x3=X[['User_id','Discount_rate','Date']].copy()
    x3 = x3.loc[x3['Date'].notnull()]
    x3.Discount_rate = map(lambda x:1-float(x.split(':')[1])/float(x.split(':')[0]) \
                             if re.match(patt,x) else float(x),x3['Discount_rate'])
    x3['user_coupons_average_discount_rate'] = x3.Discount_rate
    x3['user_coupons_max_discount_rate'] = x3.Discount_rate
    
    
    x3 = x3.groupby('User_id').agg({'user_coupons_average_discount_rate':'mean'\
                                    ,'user_coupons_max_discount_rate':'max'}).reset_index()

    
    #用户使用的满减优惠券满的平均值
    #用户使用的满减优惠券满的最大值
    #用户使用的满减优惠券减的平均值
    #用户使用的满减优惠券减的最小值
    x4=X[['User_id','Discount_rate','Date']].copy()
    x4 = x4.loc[x4['Date'].notnull()]
    x4['user_coupons_average_full']=-1
    x4['user_coupons_max_full']=-1
    x4['user_coupons_average_discount']=-1
    x4['user_coupons_min_discount']=-1
    x4.user_coupons_average_full = map(lambda x:float(x.split(':')[0]) \
                                       if re.match(patt,x) else -1,x4['Discount_rate'])
    
    x4.user_coupons_average_discount = map(lambda x:float(x.split(':')[1]) \
                                           if re.match(patt,x) else -1,x4['Discount_rate'])
    x4.user_coupons_max_full = x4.user_coupons_average_full
    x4.user_coupons_min_discount = x4.user_coupons_average_discount
    
    x4 = x4.loc[x4['user_coupons_average_full']!=-1]

    x4 = x4.groupby('User_id').agg({'user_coupons_average_full':'mean'\
                                    ,'user_coupons_average_discount':'mean'\
                                    ,'user_coupons_max_full':'max'\
                                    ,'user_coupons_min_discount':'min'\
                                    }).reset_index()
    #x4.columns=['id','min_dis','a_dis','a_full','max_full']

    #用户周末领取的优惠券被使用数/用户周末领取的优惠券
    x5=X[['User_id','Date_received','Date']].copy()
    x5['is_weekend'] = 0
    x5.is_weekend = x5.Date_received.astype('str').apply(is_weekend)
    x5 = x5.loc[x5['is_weekend']==1]
    x5['is_weekend_used']=map(lambda x:1 if x>0 else 0,x5['Date'])
    x5 = x5.groupby('User_id').agg({'is_weekend':'sum'\
                                    ,'is_weekend_used':'sum'}).reset_index()

    x5['user_weekend_coupons_use_rate'] = x5.is_weekend_used/x5.is_weekend
    x5 = x5[['User_id','user_weekend_coupons_use_rate']]

    #用户使用优惠券离商家平均距离
    #用户使用优惠券离商家最长距离
    x6 = X[['User_id','Distance','Date']].copy()
    x6 = x6.loc[x6['Date'].notnull()]
    x6['user_coupons_average_distance']=x6['Distance']
    x6['user_coupons_max_distance']=x6['Distance']
    x6 = x6.groupby('User_id').agg({'user_coupons_average_distance':'mean'\
                                    ,'user_coupons_max_distance':'max'}).reset_index()
    x6 = x6[['User_id','user_coupons_average_distance','user_coupons_max_distance']]

    #用户使用优惠券是打折类型的数量/用户使用优惠券数量
    x7=X[['User_id','Discount_rate','Date']].copy()
    x7 = x7.loc[x7['Date'].notnull()]
    x7['percent_num']=0
    x7['discount_num']=1
    x7.percent_num = map(lambda x:0 if re.match(patt,x) else 1,x7['Discount_rate'])
    
    x7 = x7.groupby('User_id').agg({'percent_num':'sum'\
                                    ,'discount_num':'sum'\
                                    }).reset_index()
    
    x7['user_coupons_is_percent_rate'] = x7.percent_num/x7.discount_num
    x7 = x7[['User_id','user_coupons_is_percent_rate']]


    user_feature = pd.merge(x1,x2,on='User_id',how='left')
    user_feature = pd.merge(user_feature,x3,on='User_id',how='left')
    user_feature = pd.merge(user_feature,x4,on='User_id',how='left')
    user_feature = pd.merge(user_feature,x5,on='User_id',how='left')
    user_feature = pd.merge(user_feature,x6,on='User_id',how='left')
    user_feature = pd.merge(user_feature,x7,on='User_id',how='left')
    

    #Merchant
    #商户被使用优惠券数/商户被领取优惠券数
    x0=X[['Merchant_id']].copy()
    x0['received_count'] = 1
    x0 = x0.groupby('Merchant_id').agg('sum').reset_index()

    x1 = X[['Merchant_id','Date']].copy()
    x1 = x1.loc[x1['Date'].notnull()]
    x1['use_count'] = 1
    x1 = x1.groupby('Merchant_id')['use_count'].agg('sum').reset_index()
    x1 = pd.merge(x0,x1,on='Merchant_id',how=('left'))
    x1['merchant_coupons_use_rate'] = x1.use_count/x1.received_count
    x1 = x1[['Merchant_id','merchant_coupons_use_rate']]
    

    #15日内商户被使用优惠券数/商户被领取优惠券数
    x2 = X[['Merchant_id','Date','Date_received']].copy()
    x2 = x2.loc[x2['Date'].notnull()]
    x2.Date = x2.Date.astype('str')
    x2.Date_received = x2.Date_received.astype('str')
    x2['Date_gap'] = x2.Date+'-'+x2.Date_received
    x2.Date_gap = x2.Date_gap.apply(get_date_gap)
    x2.Date_gap = x2.Date_gap.apply(lambda x:str(x) if x<=15 else '')
    x2 = x2.groupby('Merchant_id').agg({'Date_gap':lambda x:len(filter(None,x))}).reset_index()
    x2.rename(columns={'Date_gap':'merchant_15day_use_count'},inplace=True)
    x2 = pd.merge(x0,x2,on='Merchant_id',how=('left'))
    x2['merchant_15day_coupns_use_rate'] = x2.merchant_15day_use_count/x2.received_count
    x2 = x2[['Merchant_id','merchant_15day_coupns_use_rate']]

    #商户被使用优惠券平均优惠幅度
    #商户被使用优惠券最小优惠幅度(max)
    x3=X[['Merchant_id','Discount_rate','Date']].copy()
    x3 = x3.loc[x3['Date'].notnull()]
    x3.Discount_rate = map(lambda x:1-float(x.split(':')[1])/float(x.split(':')[0]) \
                             if re.match(patt,x) else float(x),x3['Discount_rate'])
    x3['merchant_coupons_average_discount_rate'] = x3.Discount_rate
    x3['merchant_coupons_max_discount_rate'] = x3.Discount_rate
    
    
    x3 = x3.groupby('Merchant_id').agg({'merchant_coupons_average_discount_rate':'mean'\
                                    ,'merchant_coupons_max_discount_rate':'max'}).reset_index()
    #x3.columns=['id','avg_dis_rate','max_dis_rate']
    
    #商户被使用的满减优惠券满的平均值
    #商户被使用的满减优惠券满的最大值
    #商户被使用的满减优惠券减的平均值
    #商户被使用的满减优惠券减的最小值
    x4=X[['Merchant_id','Discount_rate','Date']].copy()
    x4 = x4.loc[x4['Date'].notnull()]
    x4['merchant_coupons_average_full']=-1
    x4['merchant_coupons_max_full']=-1
    x4['merchant_coupons_average_discount']=-1
    x4['merchant_coupons_min_discount']=-1
    x4.merchant_coupons_average_full = map(lambda x:float(x.split(':')[0]) \
                                       if re.match(patt,x) else -1,x4['Discount_rate'])
    
    x4.merchant_coupons_average_discount = map(lambda x:float(x.split(':')[1]) \
                                           if re.match(patt,x) else -1,x4['Discount_rate'])
    x4.merchant_coupons_max_full = x4.merchant_coupons_average_full
    x4.merchant_coupons_min_discount = x4.merchant_coupons_average_discount
    
    x4 = x4.loc[x4['merchant_coupons_average_full']!=-1]

    x4 = x4.groupby('Merchant_id').agg({'merchant_coupons_average_full':'mean'\
                                    ,'merchant_coupons_max_full':'max'\
                                    ,'merchant_coupons_average_discount':'mean'\
                                    ,'merchant_coupons_min_discount':'min'\
                                    }).reset_index()
    
    #x4.columns=['id','max_full','a_full','a_dis','min_dis']

    #商户周末发行的优惠券被使用数/商户周末发行的优惠券
    x5=X[['Merchant_id','Date_received','Date']].copy()
    x5['is_weekend'] = 0
    x5.is_weekend = x5.Date_received.astype('str').apply(is_weekend)
    x5 = x5.loc[x5['is_weekend']==1]
    x5['is_weekend_used']=map(lambda x:1 if x>0 else 0,x5['Date'])
    x5 = x5.groupby('Merchant_id').agg({'is_weekend':'sum'\
                                    ,'is_weekend_used':'sum'}).reset_index()

    x5['merchant_weekend_coupons_use_rate'] = x5.is_weekend_used/x5.is_weekend
    x5 = x5[['Merchant_id','merchant_weekend_coupons_use_rate']]

    #商户被使用优惠券离用户平均距离
    #商户被使用优惠券离用户最长距离
    x6 = X[['Merchant_id','Distance','Date']].copy()
    x6 = x6.loc[x6['Date'].notnull()]
    x6['merchant_coupons_average_distance']=x6['Distance']
    x6['merchant_coupons_max_distance']=x6['Distance']
    x6 = x6.groupby('Merchant_id').agg({'merchant_coupons_average_distance':'mean'\
                                    ,'merchant_coupons_max_distance':'max'}).reset_index()
    x6 = x6[['Merchant_id','merchant_coupons_average_distance','merchant_coupons_max_distance']]

    #商户被使用优惠券是打折类型的数量/商户被使用优惠券数量
    x7=X[['Merchant_id','Discount_rate','Date']].copy()
    x7 = x7.loc[x7['Date'].notnull()]
    x7['percent_num']=0
    x7['discount_num']=1
    x7.percent_num = map(lambda x:0 if re.match(patt,x) else 1,x7['Discount_rate'])
    
    x7 = x7.groupby('Merchant_id').agg({'percent_num':'sum'\
                                    ,'discount_num':'sum'\
                                    }).reset_index()
    
    x7['merchant_coupons_is_percent_rate'] = x7.percent_num/x7.discount_num
    x7 = x7[['Merchant_id','merchant_coupons_is_percent_rate']]

    merchant_feature = pd.merge(x1,x2,on='Merchant_id',how='left')
    merchant_feature = pd.merge(merchant_feature,x3,on='Merchant_id',how='left')
    merchant_feature = pd.merge(merchant_feature,x4,on='Merchant_id',how='left')
    merchant_feature = pd.merge(merchant_feature,x5,on='Merchant_id',how='left')
    merchant_feature = pd.merge(merchant_feature,x6,on='Merchant_id',how='left')
    merchant_feature = pd.merge(merchant_feature,x7,on='Merchant_id',how='left')

    feature = X[['Coupon_id','User_id','Merchant_id','Date']]
    feature = pd.merge(feature,coupon_feature,on='Coupon_id',how='left')
    feature = pd.merge(feature,user_feature,on='User_id',how='left')
    feature = pd.merge(feature,merchant_feature,on='Merchant_id',how='left')
    feature = feature.drop('Coupon_id',axis=1)
    feature = feature.drop('User_id',axis=1)
    feature = feature.drop('Merchant_id',axis=1)
    label=map(lambda x:1 if x >0 else 0,feature['Date'])
    feature = feature.drop('Date',axis=1)
    print feature.info()
    imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0, copy = True)
    imputer = imputer.fit(feature)
    feature = imputer.transform(feature)
    
    return feature,label

def output_data(data,path):
    if shape(shape(data)) == (1,):
        with open(path,'w') as output:
            for i in range(len(data)):
                output.write(str(data[i])+"\n")
    elif shape(shape(data)) == (2,):
        with open(path,'w') as output:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    output.write(str(data[i][j])+" ")
                output.write("\n")

def load_data(path):
    fp=open(path,'r')
    data_list = fp.readlines()
    truth_value = []
    data = []
    len_fea = len(data_list[0].split())
    for item in data_list:
        for j in range(len_fea):
            data.append(float(item.strip().split()[j]))
    return array(data).reshape(len(data_list),len(data)/len(data_list))
def classify(min_distance_index,label):
    feat1_n = 0
    feat2_n = 0
    
    for i in range(len(min_distance_index)):
        if label[min_distance_index[i]] == 1:
            feat1_n += 1
        else:
            feat2_n += 1
    
    return 1.0*feat1_n/(feat1_n+feat2_n)
        

def get_min_distance_index(x0,data,k):
    min_distance_index = []
    for i in range(k):
        min_distance_index.append(i)
    
    distance = sum((data-x0)**2,axis=1)
    k_distance = {}.fromkeys(distance).keys()
    sorted_distance = sorted(k_distance)
    num = 0
    result =[]
    for i in range(len(sorted_distance)):
        result += where(distance==sorted_distance[i])
        num += len(result[i])
        if num >= k:
            break

    return hstack(result)


def one_scale(data):

    feat_min_value = []
    feat_max_minus_min_value = []
    
    for l in range(len(data[0])):
        feat_min_value.append(data[:,l].min())
        feat_max_minus_min_value.append(data[:,l].max() - data[:,l].min())
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = 1.0*(data[i][j] - feat_min_value[j])/feat_max_minus_min_value[j]
    
    return data
def bagging(data,S):
    data_train_index_array = []
    data_test_index_array = []
    for i in range(S):
        data_index=map(lambda x:random.randint(len(data)),range(len(data)))
        data_train_index_array.append(data_index)
        data_test_index = delete(range(len(data)),data_index,axis=0)
        data_test_index_array.append(data_test_index)

    return data_train_index_array,data_test_index_array
def handle_data():
    data,label = dataset_load()
    
    data1=[]
    data2=[]
    for i in range(len(data)):
        if label[i] == 1:
            data1.append(data[i])
        else:
            data2.append(data[i])
            
    
    data = one_scale(data)
    
    data1 = data[:len(data1)]
    data2 = data[len(data1):len(data)]
    
    output_data(data1,'data1.txt')
    output_data(data2,'data2.txt')
def draw():
    S = 15
    K = 3
    data1 =  load_data('data1.txt')
    data2 =  load_data('data2.txt')
    print 'Positive sample:',len(data1),'Negative sample:',len(data2)
    
    data = append(data1,data2,axis=0)
    label = append(array(exp(data1*0)[:,0]),array(data2*0)[:,0])

    
    
    data_train,data_test = bagging(data,S)
    
    
    data_train_array = []
    label_train_array = []
    data_test_array = []
    label_test_array = []
    for i in range(len(data_train)):
        data_train_array.append(data[data_train[i]])
        label_train_array.append(label[data_train[i]])
    for i in range(len(data_test)):
        data_test_array.append(data[data_test[i]])
        label_test_array.append(label[data_test[i]])
    for i in range(len(data_train_array)):
        print "Train data:",len(data_train_array[i]),"Test data:",len(data_test_array[i])
    
    classify_array = []

    for i in range(S):
        data0 = data_train_array[i]
        label0 = label_train_array[i]
        c=[]
        for j in range(len(data_test_array[i])):
            x0=data_test_array[i][j]
            min_distance_index = get_min_distance_index(x0,data0,K)
            c.append(classify(min_distance_index,label0))
        print i,"-",S             
        
        classify_array.append(c)
    
    final_classify_array = [0 for i in range(len(data))]
    for i in range(S):
        for j in range(len(classify_array[i])):
            final_classify_array[data_test[i][j]] = (final_classify_array[data_test[i][j]]+classify_array[i][j])/2
            
    
    
    print c_[label,final_classify_array]
    output_data(c_[label,final_classify_array],'data_classify.txt')

def ROC():
    label_forcast =  load_data('data_classify.txt')
    label = label_forcast[:,0]
    forecast = label_forcast[:,1]
    fpr,tpr,thresholds  =  roc_curve(label,forecast) 
    
    roc_auc = auc(fpr,tpr)
    plt.text(0.6,0.5,'AUC={:.6f}'.format(roc_auc))
    plt.plot([0,1],[0,1],'r+--')
    plt.plot(fpr,tpr)
    plt.title('ROC')
    plt.grid(True)
    plt.show()
handle_data()
draw()
ROC()
