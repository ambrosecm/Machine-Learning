#!/usr/bin/env python
# coding: utf-8

# In[221]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn import metrics

data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\movement_libras.data', header=None)

c_data = np.array(data.iloc[0: ,0:89])
# c_label = np.array(data.iloc[0: ,18])

print(data)


# In[223]:


c2_data=c_data[0: ,0:2]



km = KMeans(n_clusters=3).fit(c2_data)
# 标签结果
rs_labels = km.labels_
# 每个类别的中心点
rs_center_ids = km.cluster_centers_
 
featureList=[0,1]
mdl = pd.DataFrame.from_records(c2_data, columns=featureList)
print("查看KMeans聚类后的质心点，即聚类中心")
print(km.cluster_centers_) # 以数组形式查看KMeans聚类后的质心点，即聚类中心。
mdl['label'] = km.labels_ # 对原数据表进行类别标记
c = mdl['label'].value_counts()
print("原数据表进行类别标记")
print(mdl.values)# 以数组形式打印结果

# 描绘各个点
plt.scatter(c2_data[:, 0], c2_data[:, 1], c=rs_labels, alpha=0.5)
# 描绘质心
plt.scatter(rs_center_ids[:, 0], rs_center_ids[:, 1], c='red')

plt.show()


# In[226]:


c2_data=c_data[0: ,1:3]

km = KMeans(n_clusters=3).fit(c2_data)
# 标签结果
rs_labels = km.labels_
# 每个类别的中心点
rs_center_ids = km.cluster_centers_
 
featureList=[0,1]
mdl = pd.DataFrame.from_records(c2_data, columns=featureList)
print("查看KMeans聚类后的质心点，即聚类中心")
print(km.cluster_centers_) # 以数组形式查看KMeans聚类后的质心点，即聚类中心。
mdl['label'] = km.labels_ # 对原数据表进行类别标记
c = mdl['label'].value_counts()
print("原数据表进行类别标记")
print(mdl.values)# 以数组形式打印结果

# 描绘各个点
plt.scatter(c2_data[:, 0], c2_data[:, 1], c=rs_labels, alpha=0.5)
# 描绘质心
plt.scatter(rs_center_ids[:, 0], rs_center_ids[:, 1], c='red')

plt.show()


# In[228]:


c2_data=c_data[0: ,20:22]

km = KMeans(n_clusters=3).fit(c2_data)
# 标签结果
rs_labels = km.labels_
# 每个类别的中心点
rs_center_ids = km.cluster_centers_
 
featureList=[0,1]
mdl = pd.DataFrame.from_records(c2_data, columns=featureList)
print("查看KMeans聚类后的质心点，即聚类中心")
print(km.cluster_centers_) # 以数组形式查看KMeans聚类后的质心点，即聚类中心。
mdl['label'] = km.labels_ # 对原数据表进行类别标记
c = mdl['label'].value_counts()
print("原数据表进行类别标记")
print(mdl.values)# 以数组形式打印结果

# 描绘各个点
plt.scatter(c2_data[:, 0], c2_data[:, 1], c=rs_labels, alpha=0.5)
# 描绘质心
plt.scatter(rs_center_ids[:, 0], rs_center_ids[:, 1], c='red')

plt.show()


# In[172]:


X=c2_data
inertia=[]
calinski_harabaz_score=[]
a=2
for i in range(2,10):
    km = KMeans(n_clusters=i,n_init=10,init='k-means++').fit(X)
    y_pred=km.predict(X)
    center_=km.cluster_centers_
    inertia.append([i,km.inertia_])
    z=metrics.calinski_harabaz_score(X, y_pred) 
    calinski_harabaz_score.append([i,z])
    a=a+1
    plt.scatter(X[:,0],X[:,1],c=y_pred)
    plt.scatter(center_[:,0],center_[:,1],color='red')
    plt.title('n_clusters=%s'%i)
    plt.show()
plt.show()


# In[173]:


inertia=np.array(inertia)
plt.plot(inertia[:, 0], inertia[:, 1])
plt.title('SSE - n_clusters')


# In[202]:


calinski_harabaz_score=np.array(calinski_harabaz_score)
plt.plot(calinski_harabaz_score[:, 0], calinski_harabaz_score[:, 1])
plt.title('calinski_harabaz_score - n_clusters')
plt.show()


# In[174]:


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import copy
pca = PCA()  
pca = PCA(n_components = None,copy = True,whiten = False)
pca.fit(c_data)
pca.components_ 
pca.explained_variance_ratio_ 
pca = PCA(2)  #观察主成分累计贡献率,重新建立PCA模型
pca.fit(c_data)
c_data2 = pca.transform(c_data) 


# In[175]:


km = KMeans(n_clusters=3).fit(c_data2)
# 标签结果
rs_labels = km.labels_
# 每个类别的中心点
rs_center_ids = km.cluster_centers_

featureList=[0,1]
mdl = pd.DataFrame.from_records(c_data2, columns=featureList)
print("查看KMeans聚类后的质心点，即聚类中心")
print(km.cluster_centers_) # 以数组形式查看KMeans聚类后的质心点，即聚类中心。
mdl['label'] = km.labels_ # 对原数据表进行类别标记
c = mdl['label'].value_counts()
print("原数据表进行类别标记")
print(mdl.values)# 以数组形式打印结果

# 描绘各个点
plt.scatter(c_data2[:, 0], c_data2[:, 1], c=rs_labels, alpha=0.5)
# 描绘质心
plt.scatter(rs_center_ids[:, 0], rs_center_ids[:, 1], c='red')

plt.show()


# In[176]:


xx=c_data2
inertia=[]
calinski_harabaz_score=[]
a=2
for i in range(2,10):
    km = KMeans(n_clusters=i,n_init=10,init='k-means++').fit(xx)
    y_pred=km.predict(xx)
    center_=km.cluster_centers_
    inertia.append([i,km.inertia_])
    z=metrics.calinski_harabaz_score(xx, y_pred) 
    calinski_harabaz_score.append([i,z])
    a=a+1
    plt.scatter(xx[:,0],xx[:,1],c=y_pred)
    plt.scatter(center_[:,0],center_[:,1],color='red')
    plt.title('n_clusters=%s'%i)
    plt.show()
plt.show()


# In[177]:


inertia=np.array(inertia)
plt.plot(inertia[:, 0], inertia[:, 1])
plt.title('SSE - n_clusters')


# In[203]:


calinski_harabaz_score=np.array(calinski_harabaz_score)
plt.plot(calinski_harabaz_score[:, 0], calinski_harabaz_score[:, 1])
plt.title('calinski_harabaz_score - n_clusters')
plt.show()


# In[220]:


X=c_data2
#DBSCAN聚类实践
from sklearn.cluster import DBSCAN
db = DBSCAN(eps = 0.2, min_samples = 4)
db.fit(X)

db_labels = db.labels_

# 描绘各个点
plt.scatter(X[:, 0], X[:, 1], c=db_labels, alpha=0.5)


# In[205]:


# 构建空列表，用于保存不同参数组合下的结果
res = []
# 迭代不同的eps值
for eps in np.arange(0.001,1,0.05):
    # 迭代不同的min_samples值
    for min_samples in range(2,10):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        # 模型拟合
        dbscan.fit(X)
        # 统计各参数组合下的聚类个数（-1表示异常点）
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # 异常点的个数
        outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
        # 统计每个簇的样本个数
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats})
# 将迭代后的结果存储到数据框中        
df = pd.DataFrame(res)

# 根据条件筛选合理的参数组合
df.loc[df.n_clusters == 3, :]


# In[ ]:




