
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
def proDataK(df_train_GD,s,k,fillvaluemean=5.5):
    
#     fill_value = 5.8
    
    fill_value = fillvaluemean 
    
    tmp = []
    curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
    lasttime = curtime - datetime.timedelta(days = k)
    # 日期到字符串
    lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
    curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
    tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
    
    
    if len(tmp)<(k+1):
        tmp.extend(tmp[len(tmp)-(k+1):])
    
    if len(tmp) < (k+1):
        for i in range(k+1-len(tmp)):
            tmp.append(fill_value)
    
    mean = np.mean(tmp)
    if math.isnan(np.mean(tmp)): mean = 0
    tmp.append(mean)
    
    # 增加一维数据，标准差
    stdcor = np.std(tmp)
    if math.isnan(stdcor): stdcor = 0    
    tmp.append(stdcor)

    # 增加最值特征
    minvalue = np.min(tmp)
    if math.isnan(minvalue): minvalue = 0    
    tmp.append(minvalue)
    
    maxvalue = np.max(tmp)
    if math.isnan(maxvalue): maxvalue = 0    
    tmp.append(maxvalue)

    
    return tmp


def yesterdayYear(df_train_GD,s,k,fillvaluemean=5.5):
    
#     fill_value = 5.8
    fill_value = fillvaluemean 

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
    
    if len(tmp) < 2 * (k+1):
        for i in range(2 * (k+1) - len(tmp)):
            tmp.append(fill_value)
    
    mean = np.mean(tmp)
    if math.isnan(np.mean(tmp)): mean = 0
    tmp.append(mean)
    
    # 增加一维数据，标准差
    stdcor = np.std(tmp)
    if math.isnan(stdcor): stdcor = 0    
    tmp.append(stdcor)
    
    # 增加最值特征
    minvalue = np.min(tmp)
    if math.isnan(minvalue): minvalue = 0    
    tmp.append(minvalue)
    
    maxvalue = np.max(tmp)
    if math.isnan(maxvalue): maxvalue = 0    
    tmp.append(maxvalue)


    return tmp
    


# In[5]:

def getLineModel(AreaName):
    df_train_GD = df_train[df_train['地区'] == AreaName]
    
    fillvaluemean = np.mean(df_train_GD['价格'].values)
    
    year = 2016
    x = []
    y = []
    # --------- 这儿修改参数可以减少训练数据集
    kd = 10 # 表示从1-9开始的三个的数据，一共9个测试数据
    for j in range(1,kd):
        for i in range(j,j+3):
            for d in range(1,32):
                s = ''
            
                if i == 2 and d > 28:
                    continue
                
                if (i == 4 or i == 6 or i == 9 or i == 11) and d > 30:
                    continue
                
                tmp = []
                
                if i < 10:
                    s = s + str(year)+'-0'+str(i)
                else :
                    s = s + str(year)+'-'+str(i)
                
                if d < 10: 
                    s = s + '-0' + str(d)
                else: 
                    s = s + '-' + str(d)

#                 print(s)
                if len(df_train_GD[df_train_GD.index == s]['价格'].values) == 0:
                    if len(y) == 0:
                        y.append(df_train_GD[df_train_GD.index == '2015-01-01']['价格'].values[0])
                    else: # 没有这个值，用前一个代替
                        y.append(y[len(y)-1])
                else:
                    y.append(df_train_GD[df_train_GD.index == s]['价格'].values[0])


                # 添加前d天的数据特征
                for k in [1,3,5,7]:
    #                 print(proDataK(df_train_GD,s,k))
                    tmp.extend(proDataK(df_train_GD,s,k))

    #             print(len(tmp))

                # 得到前两年对应时间的前5天，后五天
                s = ''
                if i < 10:
                    s = s + str(year)+'-0'+str(i)
                else :
                    s = s + str(year)+'-'+str(i)
                
                if d < 10: 
                    s = s + '-0' + str(d)
                else: 
                    s = s + '-' + str(d)

                for k in [1,3,5,7]:
                    tmp.extend(yesterdayYear(df_train_GD,s,k))



                # 得到前一年对应时间的前5天，后五天
                s = ''
                if i < 10:
                    s = s + str(year)+'-0'+str(i)
                else :
                    s = s + str(year)+'-'+str(i)
                
                if d < 10: 
                    s = s + '-0' + str(d)
                else: 
                    s = s + '-' + str(d)

                for k in [1,3,5,7]:
                    tmp.extend(yesterdayYear(df_train_GD,s,k)) 

    #             print(tmp)
                x.append(tmp)
    
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

def Predicted(df_data,AreaName,regr, year_,month):
    '''
    df_data: 原始数据集
    AreaName ： 预测的地区
    regr: 回归函数
    year ：预测的年份
    month: 预测三个月的起始月份
    '''
    df_train_GD = df_data[df_data['地区'] == AreaName]    
    year = year_
    
    # 设置起始和结束的日期
    
    if month < 10:
        start = str(year) +'-0' +str(month) + '-01'
        end = str(year) + '-0' + str(month+2) + '-31'
    else:
        start = str(year) +'-' +str(month) + '-01'
        end = str(year) + '-' + str(month+2) + '-31'
    
#     print(start,end)
    
    x = []
    y = []
    
    y_ = 5.5
    for i in range(month,month+3):
        for d in range(1,32):
            ts = ''
            if i == 2 and d > 28: 
                continue
            if (i == 4 or i == 6 or i == 9 or i == 11) and d > 30:
                continue
                
            tmp = []
                
            if i < 10:
                ts = ts + str(year)+'-0'+str(i)
            else :
                ts = ts + str(year)+'-'+str(i)
                
            if d < 10: 
                ts = ts + '-0' + str(d)
            else: 
                ts = ts + '-' + str(d)
            
#             print(ts)
            
            # 添加前d天的数据特征
            for k in [1,3,5,7]:
                tmp.extend(proDataK(df_train_GD,ts,k))  
            
            # 得到前两年对应时间的前5天，后五天
            
            s = ''
            if i < 10:
                s = s + str(year)+'-0'+str(i)
            else :
                s = s + str(year)+'-'+str(i)
                
            if d < 10: 
                s = s + '-0' + str(d)
            else: 
                s = s + '-' + str(d)
           
            
            for k in [1,3,5,7]:
                tmp.extend(yesterdayYear(df_train_GD,s,k))

            # 得到前一年对应时间的前5天，后五天
            s = ''
            if i < 10:
                s = s + str(year)+'-0'+str(i)
            else :
                s = s + str(year)+'-'+str(i)
                
            if d < 10: 
                s = s + '-0' + str(d)
            else: 
                s = s + '-' + str(d)

            
            for k in [1,3,5,7]:
                tmp.extend(yesterdayYear(df_train_GD,s,k))
                 
    #         print(df_train_GD[lasttime:curtime])
    #         print(df_train_GD[curtime:nexttime])

    #         print(tmp)
            tmp = np.array(tmp)
            
            if y_ > 1 and y_ < 18: 
                yp_ = y_

            y_ = regr.predict(tmp)
            
            if y_ < 1 or y_ > 18: # 不满足条件设置为前一个的值
                y_ = yp_
            
            t = pd.DataFrame({'日期':ts, '地区':AreaName,'价格':y_,'数量':None,'均重':None})
            t.index = t['日期']

            df_train_GD = df_train_GD.append(t)
            
#         print(len(df_train_GD['2017-01-01':'2017-04-01']))
            
    return df_train_GD[start:end]


# In[7]:

def getValData():
    trueData = df_train['2016-10-01':'2016-12-31']
    trueData = trueData.sort(['日期','地区'])[['地区','价格']]
    
    Areas = Area
    Areas.sort()
    
    addRow = []
    indexRow = []
    fillvalue = 4
    i = j = 0
    length = len(trueData)
    while i < length:            
        row = trueData.ix[i]
        
        if str(row['地区']) != Areas[j%len(Areas)]:# 这一天没有这个地区的数据，添加这个数据，价格用填充值替代，fillvalue
            # 纪录没有出现的值，之后添加
            addRow.append([Areas[j%len(Areas)],fillvalue])  
            indexRow.append(trueData.index[i])
        else:
            i += 1

        j += 1
    
#     print(len(trueData))
    
    t = pd.DataFrame(addRow,columns=['地区','价格'],index=indexRow)
    trueData = trueData.append(t)
    
    print(len(trueData))
    return trueData

# 评价函数
def evalution(predictData, trueData):
        
    # 填充没有出现过的本应该有的数据    
    source = np.mean( np.abs(np.array(trueData['价格'].values) - np.array(predictData['价格'].values)) / trueData['价格'].values )
    print('This model sourec is', source)
    
    return source


# 验证函数,测试得分
def validation():
    '''
    使用2016-10-01 ～ 2016-12-31作为验证集
    '''
    droplist = []
    for m in [10,11,12]:
        for d in range(32):
            if m == 11 and d > 30: continue
            
            if d < 10:
                s = '2016-'+str(m)+'-0'+str(d)
            else:
                s = '2016-'+str(m)+'-'+str(d)
            
            droplist.append(s)
    
    df_train_copy = df_train.copy()
    # 得到真实数据后，删除数据，因为预测时基于了新得到的数据，
    # 为了和测试一致，这个需要删除，但是，因为预测时会再次用到原始数据，建议使用cpoy
    df_train_copy = df_train_copy.drop(droplist)
    
#     print(df_train_copy)
    
    df_preData = pd.DataFrame()
    for i in range(len(Area)):
        regrModel = getLineModel(Area[i])
        modelSet[Area[i]] = regrModel
        
        df_preData = df_preData.append(Predicted(df_train_copy,Area[i],regrModel,2016,10))
   
    
    df_preData = df_preData.sort(['日期','地区'])[['地区','价格']]
    trueData = getValData()
    
    source = evalution(df_preData,trueData)
    
    return df_preData



# In[8]:

# 验证数据,得到预测数据和真实数据
df_preData = validation()
trueData = getValData()


# In[109]:

evalution(df_preData, trueData)

print(trueData.head(15))
print(df_preData.head(15))


# In[110]:

# 在测试集上预测数据
df_ans = pd.DataFrame()

for i in range(len(Area)):
    regrModel = modelSet[Area[i]]
    df_ans = df_ans.append(Predicted(df_train,Area[i],regrModel,2017,1))


# In[111]:

# 写入文件
df_ans.info()
df_ans = df_ans.sort(['日期','地区'])[['地区','价格']]

df_ans.to_csv('data/result6.csv')


# In[ ]:



