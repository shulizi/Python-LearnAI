# coding=utf-8
#本程序可以画出ROC曲线并在图中给出AUC的值
#ROC曲线越靠近左上角，模型效果越好
#AUC（ROC曲线面积）越大，模型效果越好
def get_data(path):
    #要求数据按概率从高到低排序，不然AUC计算可能会出问题
    f = open(path)
    data_list  = f.readlines()
    return data_list
#获取概率和实际分类列表可以根据自己的测试数据文件格式自定义
def get_truth_value():#取实际分类列表
    truth_value = []
    data_list = get_data('roc.txt')
    for item in data_list:
        truth_value.append(
            item.strip().split()[2])
    return truth_value
def get_sco_list():#取概率列表
    sco_list = []
    data_list = get_data('roc.txt')
    for item in data_list:
        sco_list.append(item.strip().split()[1])
    return sco_list

def auc_calculate():#计算AUC
    #AUC公式参考 https://blog.csdn.net/qq_22238533/article/details/78666436
    truth_value = get_truth_value()
    sco_list = get_sco_list()
    rank_list = []
    same_list=[]
    n = len(truth_value)
    nump=0
    sumrankp=0
    for i in range(0,n-1):
        if  i<n-1:#未遍历到最后一个
            if len(rank_list)==i+1:#提前处理过
                if sco_list[i] == sco_list[i+1]:#概率相等的情况
                    rank_list.append(n-i-1)
                    same_list[-1][1].append(rank_list[-1])
                else: pass
            else:#未提前处理过
                if sco_list[i] == sco_list[i + 1]:#概率相等的情况
                    rank_list.append(n - i)
                    rank_list.append(n - i-1)
                    same_list.append([i, [rank_list[-2], rank_list[-1]]])
                else:
                    rank_list.append(n - i)
        else:
            if len(rank_list)<n:
                rank_list.append(n - i)
            else : pass
    for i in range(0, len(same_list)-1):
        k=same_list[i][0]
        sum=0
        for j in range(0,len(same_list[i][1])-1):
            sum+=same_list[i][1][j]
        avg=sum*1.0/len(same_list[i][1])
        for j in range(0, len(same_list[i][1]) - 1):
            rank_list[j+k]=avg
    for i in range(0, n - 1):
        if truth_value[i]=='1':
            sumrankp+=rank_list[i]
            nump += 1
    auc = (sumrankp-nump*(nump+1)*1.0/2)*1.0/(nump*(n-nump))
    return auc

def get_x_y():
    #思路参考  https://blog.csdn.net/pzy20062141/article/details/48711355
    #获取ROC曲线画图点列
    truth_value = get_truth_value()
    tpr = []
    fpr = []
    for i in range(1,len(truth_value)):
        forecast = []
        forecast.extend([1]*(i))
        forecast.extend([0]*(len(truth_value) - i))
        tmp_list = [forecast[j]*int(truth_value[j]) for j in range(len(forecast))]
        tp = sum(tmp_list[0:i])
        fp = i - tp
        fn = sum([int(x) for x in truth_value[i:len(truth_value)]])
        tn = len(truth_value)-i - fn
        tpr.append(tp*1.0/(tp+fn))
        fpr.append(fp*1.0/(tn+fp))

    return fpr,tpr
    
