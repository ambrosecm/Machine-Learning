#!/usr/bin/env python
# coding: utf-8

# In[50]:


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


data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\vehicle.data', header=None)

x_train = []                        # 存放训练集的属性
y_train = []                        # 存放训练集的标签
x_test = []                         # 存放测试集的属性
y_test = []                         # 存放测试集的标签

c_data = np.array(data.iloc[0: ,0:18])
c_label = np.array(data.iloc[0: ,18])


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(c_data, c_label,test_size=0.3)


# In[79]:


clf=LinearSVC(penalty="l2",C=1.0).fit(x_train,y_train)
y_svm_pred=clf.predict(x_test)
print('svm_confusion_matrix:')
cm=confusion_matrix(y_test,y_svm_pred)
print(cm)
print('LinearSVC_svm_classification_report:')
print(classification_report(y_test,y_svm_pred))

clf = svm.SVC(kernel='linear',C=1.0,decision_function_shape='ovo')
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print('SVC_svm_classification_report:')
print(classification_report(y_test, y_predict))

print('输出支持向量：')
print(clf.support_vectors_)


# In[53]:


#线性支持向量机准确率、召回率、f1-score可视化
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ClassificationReport
from sklearn.model_selection import train_test_split
model = LinearSVC(penalty="l2",C=1.0)
visualizer = ClassificationReport(model)
visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
visualizer.score(x_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[68]:


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import copy
pca = PCA()  
pca = PCA(n_components = None,copy = True,whiten = False)
pca.fit(x_train)
pca.components_ 
pca.explained_variance_ratio_ 
pca = PCA(2)  #观察主成分累计贡献率,重新建立PCA模型
pca.fit(x_train)
x_train2 = pca.transform(x_train) 


clf = svm.SVC(kernel='linear',C=1.0)
clf.fit(x_train2, y_train)

# plt.scatter(pca_X_test[:, 0], pca_X_test[:, 1], c=predicted3, s = 15, cmap="rainbow")


# In[78]:


for i in range(0,len(y_train)):
        if y_train[i] == ' bus ':
            plt.scatter(x_train2[i][0],x_train2[i][1],c='b',s=20)
        elif y_train[i] == ' opel':
            plt.scatter(x_train2[i][0],x_train2[i][1],c='c',s=20)
        elif y_train[i] == ' saab':
            plt.scatter(x_train2[i][0],x_train2[i][1],c='g',s=20)
        else:
            plt.scatter(x_train2[i][0],x_train2[i][1],c='g',s=20)

X=x_train2
h=1
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面
for i in range(0,len(Z)):
    if Z[i] == ' bus ':
        Z[i]=0
    elif Z[i] == ' opel':
        Z[i]=1
    elif Z[i] == ' saab':
        Z[i]=2
    else:
        Z[i]=3
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='Blues', alpha=0.5)

sv = clf.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='.',s=1)

plt.show()


# In[63]:


print(x_train2)


# In[ ]:




