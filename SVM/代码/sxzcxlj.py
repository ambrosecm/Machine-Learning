#!/usr/bin/env python
# coding: utf-8

# In[25]:


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

#文本预处理
def preprocessing(text):
    #text=text.decode("utf-8")
    tokens=[word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]  
    #nltk.sent_tokenize按句子分割 nltk.word_tokenize按词分割
    stops=stopwords.words('english')
    #stopwords停用词（is，a，an...）
    tokens=[token for token in tokens if token not in stops]
    #删除停用词
    tokens=[token.lower() for token in tokens if len(token)>=3]
    lmtzr=WordNetLemmatizer()
    tokens=[lmtzr.lemmatize(token) for  token in tokens]
    #词干提取，还原单词的一般形式
    preprocessed_text=' '.join(tokens)
    return preprocessed_text

# nltk.download('stopwords')
# nltk.download('wordnet')
#读取数据集
file_path='C:\\Users\\Administrator\\Desktop\\smsspamcollection\\SMSSpamCollection'
sms=open(file_path,'r',encoding='utf-8')
sms_data=[]
sms_label=[]
csv_reader=csv.reader(sms,delimiter='\t')
for line in csv_reader:
    sms_label.append(line[0])
    sms_data.append(preprocessing(line[1]))
sms.close()
# print(sms_data)

#按0.7：0.3比例分为训练集和测试集
dataset_size=len(sms_data)
trainset_size=int(round(dataset_size*0.7))
# print('dataset_size:',dataset_size,'trainset_size:',trainset_size)

# 将数据向量化
x_train=np.array([''.join(el) for el in sms_data[0:trainset_size]])
y_train=np.array(sms_label[0:trainset_size])

x_test=np.array(sms_data[trainset_size+1:dataset_size])
y_test=np.array(sms_label[trainset_size+1:dataset_size])

# 使用词袋模型
vectorizer=TfidfVectorizer(min_df=2,ngram_range=(1,2),stop_words='english',strip_accents='unicode',norm='l2')
# 把文本全部转换为小写，然后将文本词块化。主要是分词，分标点
X_train=vectorizer.fit_transform(x_train).toarray()
X_test=vectorizer.transform(x_test).toarray()


# In[43]:


dataSet = [0 for j in range(len(X_train))]
for i in range(len(X_train)):
    if y_train[i]=='ham':
        dataSet[i]=(1,X_train[i].tolist())
    else:
        dataSet[i]=(-1,X_train[i].tolist())
trainset_size=int(0.7*len(X_train))
datatrain=dataSet[0:trainset_size]
datatest=dataSet[trainset_size:len(X_train)]


# In[40]:


import sys;
import random;
import math;

EPS = 0.000000001  # 很小的数字，用于判断浮点数是否等于0



def svm_train(data4train, dim, W, iterations, lm, lr):
    '''
    目标函数: obj(<X,y>, W) = (对所有<X,y>SUM{max{0, 1 - W*X*y}}) + lm / 2 * ||W||^2, 即：hinge+L2
    '''
    X = [0.0 for v in range(0, dim + 1)];  # <sample, label> => <X, y>
    grad = [0.0 for v in range(0, dim + 1)];  # 梯度
    num_train = len(data4train);
    for i in range(0, iterations):
        # 每次迭代随机选择一个训练样本
        index = random.randint(0, num_train - 1);
        y = data4train[index][0];  # y其实就是label
        for j in range(0, dim + 1):
            X[j] = data4train[index][1][j];  # sample的vj

        # 计算梯度
        # for j in range(0, dim + 1):
        #	grad = lm * W[j];
        WX = 0.0;
        for j in range(0, dim + 1):
            WX += W[j] * X[j];
        if 1 - WX * y > 0:
            for j in range(0, dim + 1):
                grad[j] = lm * W[j] - X[j] * y;
        else:  # 1-WX *y <= 0的时候，目标函数的前半部分恒等于0, 梯度也是0
            for j in range(0, dim + 1):
                grad[j] = lm * W[j] - 0;

        # 更新权重, lr是学习速率
        for j in range(0, dim + 1):
            W[j] = W[j] - lr * grad[j];


def svm_predict(data4test, dim, W):
    num_test = len(data4test);
    num_correct = 0;
    for i in range(0, num_test):
        target = data4test[i][0];  # 即label
        X = data4test[i][1];  # 即sample
        sum = 0.0;
        for j in range(0, dim + 1):
            sum += X[j] * W[j];
        predict = -1;
        # print sum;
        if sum > 0:  # 权值>0，认为目标值为1
            predict = 1;
        if predict * target > 0:  # 预测值和目标值符号相同
            num_correct += 1;

    return num_correct * 1.0 / num_test;



# In[45]:


epochs = 20;  # 迭代轮数
iterations = 10;  # 每一轮中梯度下降迭代次数, 这个其实可以和epochs合并为一个参数
lm = 0.0001;  # lambda, 对权值做正则化限制的权重
lr = 0.01;  # lr, 是学习速率，用于调整训练收敛的速度
dim = 5906;  # dim, 特征的最大维度, 所有样本不同特征总数
W = [0.0 for v in range(0, dim + 1)];  # 权值

for i in range(0, epochs):
    svm_train(datatrain, dim, W, iterations, lm, lr);
    accuracy = svm_predict(datatest, dim, W);
    print("epoch:%d\taccuracy:%f" % (i, accuracy));
    # 输出结果权值
for i in range(0, dim + 1):
    if math.fabs(W[i]) > EPS:
        print("权值W%d\t%f" % (i, W[i]));


# In[ ]:




