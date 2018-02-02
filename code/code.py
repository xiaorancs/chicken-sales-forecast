
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import datetime


# In[2]:

df_train = pd.read_csv('data/trainData.csv')
df_sample = pd.read_csv('data/sample.csv')


# In[3]:

df_train.info()
df_sample.info()
# df_sample['地区'].value_counts()


# In[4]:

df_train['地区'].value_counts()
# df_train.head()
# df_train.tail()

df_train_GD = df_train[df_train['地区'] == '广东']


# In[5]:

def autoRegression(sequence, p, k=7):
    '''
    sequence ：基于时间顺序的序列
    p ：假设的参数个数，这里给出p=7
    coeff： 得到的参数
    k: 验证的个数
    检查p是否可以接受
    这里通过比较p之后的k个值，得到的预测值，是否与真实值的误差都在10^-2之内
    如果都满足就接受，否则就不接受,
    如果可以接受，就预测序列之后的值

    '''
    if len(sequence) < 2*p: return None

    x = []
    b = []
    for i in range(p):
        tmp = sequence[i:p+i]
        x.append(tmp)
    b = sequence[p:2*p]

    x = np.array(x)
    b = np.array(b)

    try: 
        if np.linalg.det(x)==0:
            return None
    except TypeError:
        return None

    coeff = np.linalg.inv(x).dot(b)
#     print(coeff)
    k = min(k, len(sequence) - 2*p)

    for i in range(k):
        t = 2*p + i
        predict = np.sum(coeff * sequence[t-p:t])
#         print(t,predict)
        if np.abs(predict - sequence[t]) > 1:
            return None


    nextvalue = np.sum(sequence[len(sequence)-p:] * coeff)

    return nextvalue


# In[6]:

k = 100
values = list(df_train_GD['价格'].values)
for i in range(90):
    nextvalue = autoRegression(values[-(k+i):],4,5)
    if(nextvalue == None): nextvalue = values[len(values)-1]
    values.append(nextvalue)
    


# In[7]:

df_train_GD.index = df_train_GD['日期']


# In[8]:

# 得到模型

# df_train_GD['2014-01-01':'2014-02-01']
# df_train_GD['2015-01-01':'2015-02-01']
# df_train_GD['2016-01-01':'2016-02-01']
year = 2016
x = []
y = []

for i in range(1,4):
    for d in range(1,32):
        if i == 2 and d > 28: continue
        tmp = []
        if d < 10: 
            s = str(year)+'-0'+str(i)+'-0'+str(d)
        else: 
            s = str(year)+'-0'+str(i)+'-'+str(d)
#         print(s)
        if len(df_train_GD[df_train_GD.index == s]['价格'].values) == 0:
            if len(y) == 0:
                y.append(df_train_GD[df_train_GD.index == '2015-01-01']['价格'].values[0])
            else:
                y.append(y[len(y)-1])
            
        else:
            y.append(df_train_GD[df_train_GD.index == s]['价格'].values[0])
        
        # 得到前一周的价格列表
        curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
        lasttime = curtime - datetime.timedelta(days = 6)
        # 日期到字符串
        lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
        curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
        tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
        
        if len(tmp)<7:
            tmp.extend(tmp[len(tmp)-7:])
        
#         print(len(tmp))
        # 得到前两年对应时间的前5天，后五天
        if d < 10: 
            s = str(year-2)+'-0'+str(i)+'-0'+str(d)
        else: 
            s = str(year-2)+'-0'+str(i)+'-'+str(d)
        curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
        lasttime = curtime - datetime.timedelta(days = 4)
        nexttime = curtime + datetime.timedelta(days = 4)
        
        curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
        lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
        nexttime = datetime.datetime.strftime(nexttime,'%Y-%m-%d')
        
        tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
        tmp.extend(list(df_train_GD[curtime:nexttime]['价格'].values))
                
        if len(tmp) < 17:
            tmp.extend(tmp[len(tmp)-17:])

        
            
        # 得到前一年对应时间的前5天，后五天
        if d < 10: 
            s = str(year-1)+'-0'+str(i)+'-0'+str(d)
        else: 
            s = str(year-1)+'-0'+str(i)+'-'+str(d)
        curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
        lasttime = curtime - datetime.timedelta(days = 4)
        nexttime = curtime + datetime.timedelta(days = 4)
        
        curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
        lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
        nexttime = datetime.datetime.strftime(nexttime,'%Y-%m-%d')
        
        tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
        tmp.extend(list(df_train_GD[curtime:nexttime]['价格'].values))
        
        if len(tmp) < 27:
            tmp.extend(tmp[len(tmp)-27:])
            
#         print(len(tmp))
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


# In[9]:

# 测试代码

print(df_train_GD[df_train_GD.index =='2016-01-01'])
print(df_train_GD['2016-01-01':'2016-01-20'])

# 得到前7天的时期
xtime = datetime.datetime.strptime('2016-01-10','%Y-%m-%d')
ytime = xtime - datetime.timedelta(days = 7)
# 日期到字符串
ytime = datetime.datetime.strftime(ytime,'%Y-%m-%d')


# In[10]:

# 得到模型

# df_train_GD['2014-01-01':'2014-02-01']
# df_train_GD['2015-01-01':'2015-02-01']
# df_train_GD['2016-01-01':'2016-02-01']
year = 2017
x = []
y = []

for i in range(1,4):
    for d in range(1,32):
        if i == 2 and d > 28: continue
        tmp = []
        if d < 10: 
            ts = str(year)+'-0'+str(i)+'-0'+str(d)
        else: 
            ts = str(year)+'-0'+str(i)+'-'+str(d)
        # 得到前一周的价格列表
        curtime = datetime.datetime.strptime(ts,'%Y-%m-%d')
        lasttime = curtime - datetime.timedelta(days = 6)
        # 日期到字符串
        lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
        curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
        tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
        
        if len(tmp)<7:
            tmp.extend(tmp[len(tmp)-7:])
        
#         print(len(tmp))
        # 得到前两年对应时间的前5天，后五天
        if d < 10: 
            s = str(year-2)+'-0'+str(i)+'-0'+str(d)
        else: 
            s = str(year-2)+'-0'+str(i)+'-'+str(d)
        curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
        lasttime = curtime - datetime.timedelta(days = 4)
        nexttime = curtime + datetime.timedelta(days = 4)
        
        curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
        lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
        nexttime = datetime.datetime.strftime(nexttime,'%Y-%m-%d')
        
        tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
        tmp.extend(list(df_train_GD[curtime:nexttime]['价格'].values))
        
#         print(df_train_GD[curtime:nexttime])
        
        if len(tmp) < 17:
            tmp.extend(tmp[len(tmp)-17:])

        
            
        # 得到前一年对应时间的前5天，后五天
        if d < 10: 
            s = str(year-1)+'-0'+str(i)+'-0'+str(d)
        else: 
            s = str(year-1)+'-0'+str(i)+'-'+str(d)
        curtime = datetime.datetime.strptime(s,'%Y-%m-%d')
        lasttime = curtime - datetime.timedelta(days = 4)
        nexttime = curtime + datetime.timedelta(days = 4)
        
        curtime = datetime.datetime.strftime(curtime,'%Y-%m-%d')
        lasttime = datetime.datetime.strftime(lasttime,'%Y-%m-%d')
        nexttime = datetime.datetime.strftime(nexttime,'%Y-%m-%d')
        
        tmp.extend(list(df_train_GD[lasttime:curtime]['价格'].values))
        tmp.extend(list(df_train_GD[curtime:nexttime]['价格'].values))
        
#         print(df_train_GD[lasttime:curtime])
#         print(df_train_GD[curtime:nexttime])
        
        if len(tmp) < 27:
            tmp.extend(tmp[len(tmp)-27:])
        
#         print(tmp)
        tmp = np.array(tmp)
        
        
        y_ = regr.predict(tmp)
        
        t = pd.DataFrame({'日期':ts, '地区':None,'价格':y_,'数量':None,'均重':None})
        t.index = t['日期']
#         print(t)
        
        df_train_GD = df_train_GD.append(t)
        
        


# In[12]:

df_train_GD['2014-12-23':'2015-01-03']
df_train_GD.tail(5)


# In[ ]:




# In[13]:

for i in range(1,4):
    for d in range(1,32):
        if i == 2 and d > 28: 
            continue            
        print(d)



# In[ ]:



