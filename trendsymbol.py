#所需函数包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import statsmodels.tsa.stattools as ts 




def trend_training(hist_data, window_length,  hts):
    #历史数据判断是否稳定
    steady_value = ts.adfuller(hist_data.values)
    if steady_value[0]<steady_value[4]['1%']:
        
#       trend ='steady'
        pass
    else:
        return '历史数据存在变化趋势'
    
    #采3sigma法估计趋势阈值h2,h3
    sigma = hist_data.values.std()
    htc = 3*sigma
    
    #仿真真趋势数据
    sim_len = len(hist_data)
    hts_unit = hts/window_length #!!!
    sim_data= pd.DataFrame(columns=['original'],data=hist_data.values, index=hist_data.index)
    sim_data['simdata']= np.ones(len(hist_data))
    sim_data['simdata'][0]=hist_data.mean()
    
    t1 = sim_data.index[0]
    for i in range(1,sim_len):
        
        t2 =sim_data.index[i]
        delta_t = (t2 - t1).seconds
        sim_data.loc[t2]['simdata']=sim_data.loc[t1]['simdata']+hts_unit*delta_t
        t1=sim_data.index[i]
    plt.figure()
    plt.plot(sim_data)
    print(sim_data.head())
    return {"window":window_length,'hts':hts_unit,'htc':htc},sim_data

    
    
    
    
def trend_identify(data,hts_unit,htc):
#     steady_value = ts.adfuller(data.values)
#     if steady_value[0]<steady_value[4]['1%']:
#         trend ='steady'
#         return trend
# 定义片段矩阵，【起始位置，值，结束位置，值】
    trend_slices=[]
    trend_names =[]
    trend_slices.append({data.index[0]:data[0],data.index[1]:data[1]})

#当前片段起始和结束时刻
    current_index_start = data.index[0]
    current_index_end = data.index[1]
    i=1
    current_index = data.index[1]
#判断最前面两个点的趋势类型：
    if abs(data[0]-data[1])<htc:

       trend_names.append(0) #0 稳定； 1 上升；-1下降；2正步；-2负步； 3上升/下降瞬变 ；-3 下降/上升瞬变

    
    while current_index<data.index[-1]:

        i+=1
        
        current_index = data.index[i]
        hts = (current_index-current_index_start).seconds * hts_unit
        #判断新点与前一片段是否连续
        Id = data[i]-list(trend_slices[-1].values())[len(trend_slices[-1])-1]
        if abs(Id)<htc: #表示新点与当前片段属于同一趋势
            current_index_end = data.index[i]
            #dict_value 取出值在python3中要强制转换为list才能继续取值。
            #表示新点与当前片段连续,当前点记入当前片段，作为最后一个点
            # 1 去除当前片段旧末尾值
            if len(trend_slices[-1])>1:
                trend_slices[-1].pop(list(trend_slices[-1].keys())[1])
            # 2 将新点作为末尾值
            trend_slices[-1][current_index]=data[i]
            #计算当前片段的最后一个值与前一片段最后一个值的差，是否>hts
            if len(trend_slices)==1:  #当只有一个片段时
                I = list(trend_slices[0].values())[1]-list(trend_slices[0].values())[0]
                if abs(I) <hts: #第一片段小于属于稳定
                    trend_names.pop()
                    trend_names.append(0)
                    trend_slices[-1][current_index_start]=data[current_index_start:current_index].mean()
                    trend_slices[-1].update({current_index_end:data[current_index_start:current_index].mean()})
                else:
                    #说明原片段起始点与当前点之间存在趋势变化：
                    #1. 确实存在趋势变化
                    #2. 原片段起始点存在较大误差，导致整体片段识别错误
                    
                    ##判断当前数据是否存在变化趋势，如不存在变化趋势，则说明起始片段定义有误，更新起始点为均值。
                    if (current_index-current_index_start).seconds>50: ##!!片段内数据量大于10=50/5个
                        steady_value = ts.adfuller(data[0:i].values)
                        if steady_value[0]<steady_value[4]['1%']:#不存在趋势
                            trend_slices[-1][current_index_start]=data[current_index_start:current_index].mean()
                            trend_slices[-1].update({current_index_end:data[current_index_start:current_index].mean()})
                        else:
                            trend_names.pop()
                            if I>0:
                                trend_names.append(1)
                            else:
                                trend_names.append(-1)
                    else:
                            trend_names.pop()
                            if I>0:
                                trend_names.append(1)
                            else:
                                trend_names.append(-1)
            else : #对于两个片段的情况
                
                I =list(trend_slices[-1].values())[len(trend_slices[-1])-1]-list(trend_slices[-2].values())[len(trend_slices[-2])-1]
                #当两个片段的末尾值相差<hts,表示当前片段段没有明前趋势变化，并判断为平稳
                if abs(I) < hts:
#                    pass
                    trend_slices[-1].update({current_index_start:data[current_index_start:current_index].mean()})
                    trend_slices[-1].update({current_index_end:data[current_index_start:current_index].mean()})
                else:
                    if (current_index-current_index_start).seconds>50:
                        steady_value = ts.adfuller(data[current_index_start:current_index].values)
                        if steady_value[0]<steady_value[4]['1%']:#不存在趋势
                            trend_slices[-1][current_index_start]=data[current_index_start:current_index].mean()
                            trend_slices[-1].update({current_index_end:data[current_index_start:current_index].mean()})
                        else:
#                            当判断存在趋势，识别如下内容，？拟合直线，斜率计算
                            
                            trend_names.pop()
                            if I>0:
                                trend_names.append(1)
                            else:
                                trend_names.append(-1)
                    else:
                        trend_names.pop()
                        if I>0:
                            trend_names.append(1)
                        else:
                            trend_names.append(-1)
        else:     #Id >htc 表示新点与当前片段不属于同一片段
            current_index_start = data.index[i]
            current_index_end = data.index[i]
            trend_slices.append({current_index_start:data[i],current_index_end:data[i]})
            trend_names.append(0)
        #判断两个片段所形成的曲线名称：步： 正步，负步  上升；下降；上升、下降瞬变；下降、上升瞬变
        if len(trend_slices)>1 and len(trend_slices[-1])>1:
            #取后两个trend_slices 定义当前片段 名称
            Is = list(trend_slices[-1].values())[1]-list(trend_slices[-1].values())[0]
            Id = list(trend_slices[-1].values())[0]-list(trend_slices[-2].values())[len(trend_slices[-2])-1]
            sign_IdIs = Id*Is>0
            trend_names[-1]= slice_identify(Is, Id,sign_IdIs,hts)
            
                    
    plt.figure()               
    for i in range(0,len(trend_slices)):
        plt.plot(data[list(trend_slices[i].keys())[0]:list(trend_slices[i].keys())[len(trend_slices[i])-1]])
        plt.plot(list(trend_slices[i].keys()),list(trend_slices[i].values()))
    return trend_slices,trend_names

def slice_identify(Is,Id,sign_IdIs,hts):
    if abs(Is)<=hts:
        if Id >0:
            name = 2
        else:
            name = -2
    else:
        if sign_IdIs:
            if Id >0:
                name=1
            else:
                name=-1
        else:
            if Id >0:
                name=3
            else:
                name = -3
                
    return name



#          if abs(data[i]-list(trend_slices[-1].values())[1])>=htc:
# =============================================================================
# 
# =============================================================================
# train data building
rng = pd.date_range('1/1/2018',periods=1000,freq='5S')
rnd = np.random.RandomState(0)
a = rnd.randn(1000)
a[0:200]=a[0:200]+np.ones(200)*2
a[200:500]=a[200:500]+np.linspace(2.01,12,300)
a[500:1001]=a[500:1001]+np.ones(500)*12
test_data = pd.Series(a,index=rng)
train_data = pd.Series(rnd.randn(1000),index=rng)

parameter , _ = trend_training(train_data,500,5)
    
trend_slice, names=trend_identify(test_data,parameter['hts'],parameter['htc'])    
print(trend_slice)
print(names)