#!/usr/bin/python
# -*- coding: utf-8 -*
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
def dataset_load():
    datas=pd.read_csv(r'mushrooms.csv')
# print(datas.isnull().sum()) 有无无效数据
    #datas=datas.loc[datas['stalk-root']!='?']
    X=datas.drop('class',axis=1)
    Y=datas['class']
#哑变量
    Encoder_X = LabelEncoder()
    for col in X.columns:
        X[col] = Encoder_X.fit_transform(X[col])
        print(col,Encoder_X.classes_)
    X=pd.get_dummies(X,columns=X.columns,drop_first=True)
    Encoder_y=LabelEncoder()
    Y = Encoder_y.fit_transform(Y)
#分割数据集
    #改test_size，这里相当于百分之60是测试集
    x_train, x_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.6, random_state=0)


    return x_train, y_train,x_test, y_test

dataset_load()
