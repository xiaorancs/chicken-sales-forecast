
# coding: utf-8

# In[15]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import datetime

import xgboost as xgb


# In[16]:

df_train = pd.read_csv('data/trainData.csv')
df_sample = pd.read_csv('data/sample.csv')

df_train.index = df_train['日期']

df_train.info()
df_sample.info()
# df_sample['地区'].value_counts()


# In[17]:

Area = set(df_sample['地区'].values)
Area = list(Area)


# In[22]:

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
    
#     from sklearn import linear_model

#     regr = linear_model.LinearRegression()
#     regr.fit(x, y)

        
    xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=10, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)

    xlf.fit(x, y)

    
    
    return xlf


# In[30]:

def Predicted(AreaName,regr):
    df_train_GD = df_train[df_train['地区'] == AreaName]
    year = 2017
    x = []
    y = []

    for i in range(1,4):
        for d in range(1,32):
            if i == 2 and d > 28: 
                continue
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


            y_ = regr.predict([tmp])
            
            print(y_)
            
            t = pd.DataFrame({'日期':ts, '地区':AreaName,'价格':y_,'数量':None,'均重':None})
            t.index = t['日期']
    #         print(t)

            df_train_GD = df_train_GD.append(t)
            
#         print(len(df_train_GD['2017-01-01':'2017-04-01']))
            
    return df_train_GD['2017-01-01':'2017-03-31']


# In[31]:

len(Area)


# In[32]:

df_ans = pd.DataFrame()

for i in range(len(Area)):
    regrModel = getLineModel(Area[i])
    df_ans = df_ans.append(Predicted(Area[i],regrModel))


# In[29]:

df_ans.info()
df_ans = df_ans.sort(['日期','地区'])[['地区','价格']]


# In[47]:

df_ans


# In[48]:

df_ans.to_csv('data/result.csv',index=False,index_label=False)


# In[53]:

# 结合预测结果，与2014-2016年对应的时间取均值
df_ans1 = df_train['2014-01-01':'2014-03-31'][['日期','地区','价格']].sort(['日期','地区'])
df_ans2 = df_train['2015-01-01':'2015-03-31'][['日期','地区','价格']].sort(['日期','地区'])
df_ans3 = df_train['2016-01-01':'2016-03-31'][['日期','地区','价格']].sort(['日期','地区'])




# In[62]:

df_ans.info()
df_ans1.info()
df_ans2.info()
df_ans3.info()

df_ans1.groupby('日期').count()



# In[ ]:




# In[ ]:



