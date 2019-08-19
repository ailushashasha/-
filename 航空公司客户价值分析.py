
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:

datafile = (r'F:\学习资料\《Python数据分析与挖掘实战》源数据和代码\C07\chapter7\demo\data\air_data.csv')
data = pd.read_csv(datafile,encoding='utf-8')
data.T


# In[22]:

explore = data.describe(percentiles=[], include='all').T
explore['null'] = len(data) - explore['count']

explore = explore[['null','max','min']]
explore.columns = ['空值数','最大值','最小值']


# In[3]:

'''
丢弃票价为Nan的记录
丢弃票价为0、平均折扣率不为0、总飞行公里数大于0的记录
'''
data = data[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()]
# 删除票价空值的记录

index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0 ) & (data['avg_discount'] == 0 )
data = data[index1 | index2 | index3]
# 删除票价为0、平均折扣率不为0、总飞行公里数大于0的记录

data.T


# In[4]:

data_new = data[['FFP_DATE','LOAD_TIME','FLIGHT_COUNT','avg_discount','SEG_KM_SUM','LAST_TO_END']]
# 筛选与建模LRFMC指标有关的数据
data_new.T


# In[5]:

data['LOAD_TIME'] = pd.to_datetime(data['LOAD_TIME'])
data['FFP_DATE'] = pd.to_datetime(data['FFP_DATE'])
data['入会时间'] = data['LOAD_TIME'] - data['FFP_DATE']
data['平均每公里票价'] = (data['SUM_YR_1'] + data['SUM_YR_2']) / data['SEG_KM_SUM']
data['时间间隔差值'] = data['MAX_INTERVAL'] - data['AVG_INTERVAL']

data_copy = data.rename(columns = {'FLIGHT_COUNT':'飞行次数','SEG_KM_SUM':'飞行总里程','avg_discount':'平均折扣率'},inplace=False)
data_new = data_copy[['入会时间','飞行次数','平均每公里票价','飞行总里程','时间间隔差值','平均折扣率']]

data_new['入会时间'] = data_new['入会时间'].astype(np.int64)/(60*60*24*10**9)
# 将时间序列转为数值类型

data_new.head()


# In[6]:

data_new_zscore = (data_new - data_new.mean()) / (data_new.std())
# 标准化
data_new_zscore.head()


# In[7]:

def distEclud(vecA,vecB):
    return(np.sum(np.power(vecA-vecB,2)))
# 两个向量的欧式距离的平方

def test_KMeans_nclusters(data_train):
    #data_train = data_train.values
    nums = range(2,10)
    SSE=[]
    for num in nums:
        sse=0
        kmodel = KMeans(n_clusters=num)
        kmodel.fit(data_train)
        cluster_center_list = kmodel.cluster_centers_
        cluster_list = kmodel.labels_.tolist()
        for index in range(len(data)):
            cluster_num = cluster_list[index]
            sse += distEclud(data_train[index,:],cluster_center_list[cluster_num])
        print('簇数是%i时'%num,'SSE是%i'%sse)
        SSE.append(sse)
    return(nums,SSE)

nums,SSE = test_KMeans_nclusters(data_new_zscore)


# In[8]:

fig = plt.figure(figsize=(8,6))
plt.plot(nums,SSE,marker="o",color='g')
plt.xlabel('n_clusters')
plt.ylabel('SSE')
plt.grid()
plt.show()


# In[26]:

kmeans = KMeans(5)
kmeans.fit(data_new_zscore)
# kmeans聚类

print(kmeans.cluster_centers_)
print(kmeans.labels_)
# 聚类中心
# 每个样本对应的类别

r1 = pd.Series(kmeans.labels_).value_counts()
r2 = pd.DataFrame(kmeans.cluster_centers_)

max = r2.values.max()
min = r2.values.min()

r = pd.concat([r2,r1],axis=1)
r.columns = list(data_new_zscore.columns) + ['kmeans类别数目']
print(r)
# 合并成一个新的dataframe


fig2 = plt.figure(figsize=(10,8))
ax = fig2.add_subplot(111,projection='polar')
center_num = r.values
labels = ['入会时间','飞行次数','平均每公里票价','飞行总里程','时间间隔差值','平均折扣率']
N = 6

for i,v in enumerate(center_num):
    angles = np.linspace(0,2*np.pi,N,endpoint=False)
    center = np.concatenate((v[:-1],[v[0]]))
    angles = np.concatenate((angles,[angles[0]]))
    # center,angles首尾闭合
    ax.plot(angles, center, 'o-', linewidth=2, label = "第%d簇人群,有%d人"% (i+1,v[-1]))
    ax.fill(angles,center,alpha=0.25)
    plt.thetagrids(angles*180/np.pi,labels)
    plt.ylim(min-0.1,max+0.1)
    plt.grid(True)
    plt.legend(loc='upper right',ncol=1,bbox_to_anchor=(1.4,1.1),shadow=True)
plt.show()


# In[ ]:



