
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import datetime


# In[2]:

df_train = pd.read_csv('data/trainData.csv')
df_sample = pd.read_csv('data/sample.csv')

df_train.index = df_train['日期']

df_train.info()
df_sample.info()
# df_sample['地区'].value_counts()


# In[3]:

Area = set(df_sample['地区'].values)
Area = list(Area)
Area.sort()
# 以城市为索引存储模型集合
modelSet = {}


# In[4]:

# 得到前k天的价格列表
def proDataK(df_train_GD,s,k):
    tmp = []
    curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
    lasttime = curtime - datetime.timedelta(days = k)
    # 日期到字符串
    lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
    curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
    tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
    
    
    if len(tmp)<(k+1):
        tmp.extend(tmp[len(tmp)-(k+1):])
    
    mean = np.mean(tmp)
    if math.isnan(np.mean(tmp)): mean = 0
    
    tmp.append(mean)
    
    return tmp


def yesterdayYear(df_train_GD,s,k):
    tmp = []
    curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
    lasttime = curtime - datetime.timedelta(days = k)
    nexttime = curtime + datetime.timedelta(days = k)

    curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
    lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
    nexttime = datetime.datetime.strftime(nexttime,'%Y-%m-%d')

    tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
    tmp.extend(list(df_train_GD[curtime:nexttime]['价格'].values))
    
    
    if len(tmp) < 2 * (k+1):
        tmp.extend(tmp[len(tmp)-2 * (k+1):])

    mean = np.mean(tmp)
    if math.isnan(np.mean(tmp)): mean = 0
    
    tmp.append(mean)
    
    return tmp
    


# In[5]:

def getLineModel(AreaName):
    df_train_GD = df_train[df_train['地区'] == AreaName]
    year = 2016
    x = []
    y = []

    for i in range(1,4):
#         print(i)
        for d in range(1,32):
            if i == 2 and d > 28:
                continue
            
            tmp = []
            if d < 10: 
                s = str(year)+'-0'+str(i)+'-0'+str(d)
            else: 
                s = str(year)+'-0'+str(i)+'-'+str(d)
                
    #         print(s)
            if len(df_train_GD[df_train_GD.index == s]['价格'].values) == 0:
                if len(y) == 0:
                    y.append(df_train_GD[df_train_GD.index == '2016-01-01']['价格'].values[0])
                else:
                    y.append(y[len(y)-1])
            else:
                y.append(df_train_GD[df_train_GD.index == s]['价格'].values[0])
            
            
            # 添加前d天的数据特征
            for k in [1,3,5,7]:
#                 print(proDataK(df_train_GD,s,k))
                tmp.extend(proDataK(df_train_GD,s,k))
            
            if len(tmp) < 24:
                tmp.extend(tmp[len(tmp)-24:])
            
            
#             print(len(tmp))
            
            
            # 得到前两年对应时间的前5天，后五天
            if d < 10: 
                s = str(year-2)+'-0'+str(i)+'-0'+str(d)
            else: 
                s = str(year-2)+'-0'+str(i)+'-'+str(d)
            
            for k in [1,3,5,7]:
                tmp.extend(yesterdayYear(df_train_GD,s,k))
            
            
            
            # 得到前一年对应时间的前5天，后五天
            if d < 10: 
                s = str(year-1)+'-0'+str(i)+'-0'+str(d)
            else: 
                s = str(year-1)+'-0'+str(i)+'-'+str(d)

            for k in [1,3,5,7]:
                tmp.extend(yesterdayYear(df_train_GD,s,k))
            
            if len(tmp) < 112:
                tmp.extend(tmp[len(tmp)-112:])
            
    
#             print(tmp)
            x.append(tmp)

    #         print(s)

    # 得到参数
    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(len(y))

    from sklearn import linear_model

    regr = linear_model.LinearRegression()
    
    regr.fit(x, y)

    regr.coef_
    
    return regr




# In[6]:

def Predicted(AreaName,regr):
    df_train_GD = df_train[df_train['地区'] == AreaName]
    year = 2017
    x = []
    y = []
    
    y_ = 5
    for i in range(1,4):
        for d in range(1,32):
            if i == 2 and d > 28: 
                continue
            tmp = []
            if d < 10: 
                ts = str(year)+'-0'+str(i)+'-0'+str(d)
            else: 
                ts = str(year)+'-0'+str(i)+'-'+str(d)
            
            # 添加前d天的数据特征
            for k in [1,3,5,7]:
                tmp.extend(proDataK(df_train_GD,ts,k))
            
            if len(tmp) < 24:
                tmp.extend(tmp[len(tmp)-24:])
            
            
    #         print(len(tmp))
            # 得到前两年对应时间的前5天，后五天
            if d < 10: 
                s = str(year-2)+'-0'+str(i)+'-0'+str(d)
            else: 
                s = str(year-2)+'-0'+str(i)+'-'+str(d)
            
            
            for k in [1,3,5,7]:
                tmp.extend(yesterdayYear(df_train_GD,s,k))

    #         print(df_train_GD[curtime:nexttime])

            # 得到前一年对应时间的前5天，后五天
            if d < 10: 
                s = str(year-1)+'-0'+str(i)+'-0'+str(d)
            else: 
                s = str(year-1)+'-0'+str(i)+'-'+str(d)
           
            
            for k in [1,3,5,7]:
                tmp.extend(yesterdayYear(df_train_GD,s,k))
            
            
            if len(tmp) < 112:
                tmp.extend(tmp[len(tmp)-112:])
                
    #         print(df_train_GD[lasttime:curtime])
    #         print(df_train_GD[curtime:nexttime])

    #         print(tmp)
            tmp = np.array(tmp)
            
            if y_ > 1 and y_ < 18: 
                yp_ = y_

            y_ = regr.predict(tmp)
            
            if y_ < 1 or y_ > 18: # 设置为前一个的值
                y_ = yp_
            
            t = pd.DataFrame({'日期':ts, '地区':AreaName,'价格':y_,'数量':None,'均重':None})
            t.index = t['日期']
    #         print(t)

            df_train_GD = df_train_GD.append(t)
            
#         print(len(df_train_GD['2017-01-01':'2017-04-01']))
            
    return df_train_GD['2017-01-01':'2017-03-31']


# In[61]:

df_ans = pd.DataFrame()

for i in range(len(Area)):
    regrModel = getLineModel(Area[i])
    df_ans = df_ans.append(Predicted(Area[i],regrModel))


# In[62]:

# 写入文件
df_ans.info()
df_ans = df_ans.sort(['日期','地区'])[['地区','价格']]

df_ans.to_csv('data/result3_1.csv')


# In[63]:

df_ans


# In[ ]:



