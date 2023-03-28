#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


#svm
clf=LinearSVC(penalty="l2",C=1.0).fit(X_train,y_train)
y_svm_pred=clf.predict(X_test)
print('svm_confusion_matrix:')
cm=confusion_matrix(y_test,y_svm_pred)
print(cm)
print('svm_classification_report:')
print(classification_report(y_test,y_svm_pred))

clf = svm.SVC(kernel='linear',C=1)
clf.fit(X_train, y_train)
print(clf.support_vectors_)
print('支持向量中非零元素所在位置：')
print(np.flatnonzero(clf.support_vectors_))
print('支持向量中非零元素：')
print(clf.support_vectors_.ravel()[np.flatnonzero(clf.support_vectors_)])


# In[6]:


#线性支持向量机准确率、召回率、f1-score可视化
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ClassificationReport
from sklearn.model_selection import train_test_split
model = LinearSVC()
visualizer = ClassificationReport(model)
visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[7]:


from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
pca = PCA()  
pca = PCA(n_components = None,copy = True,whiten = False)
pca.fit(X_train)
pca.components_ 
pca.explained_variance_ratio_ 
 
pca = PCA(2)  #观察主成分累计贡献率,重新建立PCA模型
pca.fit(X_train)
X_train2 = pca.transform(X_train) 

pca = PCA()  
pca = PCA(n_components = None,copy = True,whiten = False)
pca.fit(X_test)
pca.components_ 
pca.explained_variance_ratio_ 
 
pca = PCA(2)  #观察主成分累计贡献率,重新建立PCA模型
pca.fit(X_test)
X_test2 = pca.transform(X_test) 

clf = SVC(C=1)
clf.fit(X_train2, y_train)


def plot_support(support_vector, data, labels, clf):
    for i in range(0,len(data)):
        if labels[i] == 'ham':
            plt.scatter(data[i][0],data[i][1],c='b',s=20, label = '1')
        else:
	        plt.scatter(data[i][0],data[i][1],c='g',s=20, label = '0')
    for j in support_vector:
        plt.scatter(data[j][0], data[j][1], s = 100,c = '',
                    linewidth=1.5, edgecolor='red')
    x_tmp = np.linspace(0, 1, 500)
    y_tmp = np.linspace(0, 0.8, 500)
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    xy = np.vstack([X_tmp.ravel(), Y_tmp.ravel()]).T
    P = clf.decision_function(xy).reshape(X_tmp.shape)
    plt.contour(X_tmp, Y_tmp, P, levels=[-1,0,1],colors='k',
                linewidths=1, linestyles=["--","-","--"])
    plt.show()

support = clf.support_
plot_support(support, X_train2, y_train, clf)


# In[8]:


clf = SVC(kernel='linear',C=1)
clf.fit(X_train2, y_train)
print(clf.support_vectors_)


# In[9]:


#使用核方法
clf = SVC(kernel='rbf', C=1E6) #引入径向基 函数
clf.fit(X_train, y_train)
y_svm_pred=clf.predict(X_test)
print('svm_confusion_matrix:')
cm=confusion_matrix(y_test,y_svm_pred)
print(cm)
print('svm_classification_report:')
print(classification_report(y_test,y_svm_pred))


# In[10]:


#线性支持向量机准确率、召回率、f1-score可视化
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ClassificationReport
from sklearn.model_selection import train_test_split

clf = SVC(kernel='rbf', C=1E6) #引入径向基 函数
clf.fit(X_train, y_train)
visualizer = ClassificationReport(clf)
visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


# In[11]:


def plt_scatter( data, labels):
    for i in range(0,len(data)):
        if labels[i] == 'ham':
            plt.scatter(data[i][0],data[i][1],c='b',s=20, label = '1')
        else:
	        plt.scatter(data[i][0],data[i][1],c='g',s=20, label = '0')
def plot_svc_decision_function(model, ax=None):
    if ax is None:
        ax = plt.gca()
    # 画决策边界：制作网格，理解函数meshgrid
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #在最大值和最小值之间形成30个规律的数据
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    #使用meshgrid函数将两个一维向量转换为特征矩阵
    #核心是将两个特征向量广播，以便获取y.shape * x.shape这么多个坐标点的横坐标和纵坐标
    X,Y = np.meshgrid(x,y)
    #其中ravel()是降维函数，vstack能够将多个结构一致的一维数组按行堆叠起来
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    #重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
    #然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致
    P = model.decision_function(xy).reshape(X.shape)
    #画决策边界和平行于决策边界的超平面
    ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

clf = SVC(kernel='rbf', C=1E6) #引入径向基 函数
clf.fit(X_train2, y_train)
plt_scatter(X_train2, np.array(y_train))
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none');
plt.show()


# In[ ]:




