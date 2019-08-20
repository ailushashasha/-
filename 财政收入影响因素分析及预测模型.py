#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


inputfile = r'F:\xuexiziliao\《Python数据分析与挖掘实战》源数据和代码\Python数据分析与挖掘实战\chapter13\demo\data\data1.csv'
data_original = pd.read_csv(inputfile)
data_original = pd.DataFrame(data_original)
print(data_original)
#print(data_original.shape)
data_info = [data_original.min(),data_original.max(),data_original.mean(),data_original.std()]
data_info = pd.DataFrame(data_info,index=['min','max','mean','std'])

# 对源数据进行pearson相关性分析
data_pearson = data_original.corr(method='pearson',min_periods=1)
print(data_pearson.loc['y'])
print('p_x11_y绝对值小于0.3，不存在线性相关;\n其余r绝对值大于0.8，高度线性相关')

# 0-1标准化
data_bzh = data_original.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
#print(data_bzh)

# 将标准化后的数据作图，可以明显的看出x11与y非线性相关
fig = plt.figure(figsize=(10,5))
for i in range(0,13):
    plt.plot(data_bzh.T.iloc[i],data_bzh.T.loc['y'])
    labels = data_bzh.columns.tolist()
    plt.legend(ncol=6,bbox_to_anchor=(0.9,-0.1),labels=[labels[i]+' - y' for i in range(0,13)])
    plt.grid()


# In[3]:


#导入Adaptive Lasso算法
from sklearn.linear_model import Lasso
# sklearn 0.21版本cannot import AdaptiveLasso
# 问题暂时未得到解决，网上有说参考R的做法，就是第一步lasso回归，然后x乘以回归系数倒数的绝对值，之后再做一遍lasso等

lmodel = Lasso()
lmodel.fit(data_original.iloc[:,:13],data_original['y'])
xs = lmodel.coef_  # 特征系数 
xs = pd.Series(xs,index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13'])
print(xs)


# In[4]:


# 财政收入灰色预测模型

import sys
#添加自定义函数所在位置
sys.path.append(r'F:\xuexiziliao\数据分析师学习_python学习\GM11')
#前一个GM11为文件名，后一个为函数名
from GM11 import GM11

data_c = data_original.copy()
data_c.index = np.arange(1994,2014)
data_c.loc[2014] = None
data_c.loc[2015] = None
#print(data_c)

l=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x13']
for i in l:
    f = GM11(data_c[i][np.arange(1994, 2014)].values)[0]    #灰色预测函数
    data_c[i][2014] = f(len(data_c) - 1)    #2014年预测结果
    data_c[i][2015] = f(len(data_c))        #2015年预测结果
    data_c[i] = data_c[i].round(2)          #保留两位小数
data_c.drop(['x12'],axis=1,inplace=True)
data_c


# In[40]:


# 财政收入神经网络预测模型

data_fin = data_c
data_train = data_c.loc[np.arange(1994,2014)].copy()
data_train_mean = data_train.mean()
data_train_std = data_train.std()
data_train_bzh = (data_train - data_train_mean) / data_train_std  # 数据0均值标准化

feature = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x13']
x_train = data_train_bzh[feature].values
y_train = data_train_bzh['y'].values

from keras.models import Sequential
from keras.layers.core import Dense,Activation
# 此处要装 keras\tensorflow

kmodel = Sequential()
kmodel.add(Dense(input_dim=12,output_dim=12))
kmodel.add(Activation('relu'))   # 用relu函数作为激活函数，能够大幅提高准确率
kmodel.add(Dense(input_dim=12,output_dim=1))
kmodel.compile(loss='mean_squared_error',optimizer='adam')   #编译模型，目标函数是均方差
kmodel.fit(x_train,y_train,nb_epoch=10000,batch_size=16)   # 训练模型，学习一万次
kmodel.save_weights(r'F:\xuexiziliao\数据分析师学习_python学习\07-财政项目分析\1.model') # 保存模型


# In[89]:


# 预测还原结果

x = ((data_fin[feature] - data_train_mean[feature]) / data_train_std[feature]).values
data_fin['y_pred'] = kmodel.predict(x) * data_train_std['y'] + data_train_mean['y']
#print(data_fin)

fig2 = plt.figure(figsize=(12,8))
ax1 = fig2.add_subplot(211)
data_fin[['y','y_pred']].plot(kind='bar',rot=45,ax=ax1)

ax2 = fig2.add_subplot(212)
data_fin[['y','y_pred']].plot(style=['b-o','r-*'],rot=45,ax=ax2,xticks=(np.arange(1994,2016)))


