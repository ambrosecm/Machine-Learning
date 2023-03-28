#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
from random import sample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 标准化工具
import matplotlib.pyplot as plt


# In[72]:


data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\vehicle.data', header=None)

x_train = []                        # 存放训练集的属性
y_train = []                        # 存放训练集的标签
x_test = []                         # 存放测试集的属性
y_test = []                         # 存放测试集的标签
print(data)
c_data = np.array(data.iloc[0: ,0:18])
c_label = np.array(data.iloc[0: ,18])
x_train, x_test, y_train, y_test = train_test_split(c_data, c_label,test_size=0.2)


# In[73]:


scaler =StandardScaler()
scaler.fit(x_train)
X_train =scaler.transform(x_train)
X_test =scaler.transform(x_test)


# In[74]:


import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#激活函数tanh
def tanh(x):
    return np.tanh(x)
#tanh的导函数，为反向传播做准备
def tanh_deriv(x):
    return 1-np.tanh(x)*np.tanh(x)
#激活函数逻辑斯底回归函数
def logistic(x):
    return 1/(1+np.exp(-x))
#激活函数logistic导函数
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))
#神经网络类
class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):
    #根据激活函数不同，设置不同的激活函数和其导函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
       #初始化权重向量，从第一层开始初始化前一层和后一层的权重向量
        self.weights = []
        for i in range(1 , len(layers)-1):
         #权重的shape，是当前层和前一层的节点数目加１组成的元组
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            #权重的shape，是当前层加１和后一层组成的元组
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)
    #fit函数对元素进行训练找出合适的权重，X表示输入向量，y表示样本标签，learning_rate表示学习率
    #epochs表示循环训练次数
    def fit(self , X , y , learning_rate=0.2 , epochs=10000):
        X  = np.atleast_2d(X)#保证X是二维矩阵
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1] = X
        X = temp #以上三步表示给Ｘ多加一列值为１
        y = np.array(y)#将y转换成np中array的形式
        #进行训练
        for k in range(epochs):
            i = np.random.randint(X.shape[0])#从0-epochs任意挑选一行
            a = [X[i]]#将其转换为list
            #前向传播
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            #计算误差
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            #反向传播，不包括输出层
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            #更新权重
            for i in range(len(self.weights)):
                layer  = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate*layer.T.dot(delta)

    #进行预测
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a


# In[85]:


import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


nn = NeuralNetwork([18, 10, 4], 'logistic')
# 对标签进行标签化
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print('start fitting')
nn.fit(X_train, labels_train, epochs=30000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))  # 选择概率最大的下标作为预测结果
# 预测结果
print(predictions)
test = []
for i in range(len(labels_test)):
    for j in range(len(labels_test[i])):
        if labels_test[i][j] == 1:
            test.append(j)
# 混淆矩阵
print(confusion_matrix(test, predictions))
# 分类报告
print(classification_report(test, predictions))


# In[ ]:




