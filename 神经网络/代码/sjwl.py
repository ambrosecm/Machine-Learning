#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import sample
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier ## 导入神经网络包
from sklearn.preprocessing import StandardScaler  # 标准化工具
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn.metrics import classification_report #预测报告


# In[71]:


data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\vehicle.data', header=None)

x_train = []                        # 存放训练集的属性
y_train = []                        # 存放训练集的标签
x_test = []                         # 存放测试集的属性
y_test = []                         # 存放测试集的标签
print(data)
c_data = np.array(data.iloc[0: ,0:18])
c_label = np.array(data.iloc[0: ,18])

x_train, x_test, y_train, y_test = train_test_split(c_data, c_label,test_size=0.2)


# In[3]:


scaler =StandardScaler()
scaler.fit(x_train)
X_train =scaler.transform(x_train)
X_test =scaler.transform(x_test)


# In[61]:


mlp =MLPClassifier(
    hidden_layer_sizes=(10,10,10),
    solver='adam',
    alpha=0.0001,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[21]:


mlp =MLPClassifier(
    hidden_layer_sizes=(10,10,10),
    solver='lbfgs',
    alpha=0.0001,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[22]:


mlp =MLPClassifier(
    hidden_layer_sizes=(10,10,10),
    solver='sgd',
    alpha=0.0001,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[62]:


mlp =MLPClassifier(
    hidden_layer_sizes=(10,10),
    solver='adam',
    alpha=0.0001,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[66]:


mlp =MLPClassifier(
    hidden_layer_sizes=(12,8),
    solver='adam',
    alpha=0.0001,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[69]:


mlp =MLPClassifier(
    hidden_layer_sizes=(14,10,6),
    solver='adam',
    alpha=0.0001,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[64]:


mlp =MLPClassifier(
    hidden_layer_sizes=(10),
    solver='adam',
    alpha=0.0001,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[38]:


mlp =MLPClassifier(
    hidden_layer_sizes=(10,10,10),
    solver='adam',
    alpha=0.0005,
    max_iter=1000)
mlp.fit(X_train,y_train)
predictions=mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#线性支持向量机准确率、召回率、f1-score可视化热图
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(mlp)
visualizer.fit(X_train,y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[ ]:




