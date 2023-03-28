#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
X_test=vectorizer.transform(x_test)

d=vectorizer.vocabulary_
a=sorted(d.items(), key=lambda d:d[1])
a=dict(a)
a=list(a.keys())#a为特征值
X_train=X_train.tolist()
for i in range(0,len(X_train)):
    for j in range(0,len(X_train[i])):
        if X_train[i][j] !=0:
            X_train[i][j]=1
    X_train[i].append(y_train[i])
# print(X_train[1])


# In[6]:


import math
#定义一个求熵的函数，输入为一维数组，输出为熵。
def getShang(X):
    temp={k:X.count(k) for k in set(X)}
    s=0
    for k,v in temp.items():
        s=s+(v/len(X))*(-math.log2(v/len(X)))
    return s
#定义一个求最优特征的函数，输入为一个二维数组，输出为最优特征所在的列
def getBestIndex(X):
    T=[]
    #遍历数组X的每一列，计算每个特征列对应的熵，放到数组Shang中
    for i in range(len(X[0])-1):
        #取出数组X的每一列(除了最后一列)，定义临时数组temp1
        temp1=[x[i] for x in X]
        #遍历数组temp1，得到每个元素以及该元素出现的次数，以字典temp2来储存
        temp2={k:temp1.count(k) for k in set(temp1)}
        shang=0
        for k,v in temp2.items():
            temp3=[]
            for j in range(len(temp1)):
                if temp1[j]==k:
                    #Y的index与temp1的index相同
                    temp3.append([x[len(X[0])-1] for x in X][j])
            shang=shang+v/len(X)*getShang(temp3)
        T.append(shang)
    bestIndex=T.index(min(T))
    return bestIndex
#定义一个求决策树的函数，输入为二维数组加分类标签，输出为决策树
def fit(X,lable):
    tree={}
    #最好特征所在的位置为bestIndex
    bestIndex=getBestIndex(X)
    #最好的位置对应的特征列，记为bestList
    bestList=[x[bestIndex] for x in X]
    #对最好特征列去重，得到PureBestList
    pureBestList=list(set(bestList))
    branch={}
    tree[lable[bestIndex]]=branch
    #
    for v in pureBestList:
        #temp4用来承装利用v对X分组得到的结果
        temp4=[]
        for i in range(len(bestList)):
            if bestList[i]==v:
                temp4.append(X[i][-1])
        #如果能分完
        if getShang(temp4)==0:
            branch[v]=temp4[0]
        else:
            #对数组X划分子集subX，划分标准:在X中提取最好特征列中元素为v时所有的行
            #第一步：在X中提取最好特征列中元素为v时所有的行，得到sub1X
            sub1X=[x for x in X if x[bestIndex]==v]
            #第二步：删除最好特征列，得到subX
            subX=[[x[i] for i in range(len(x)) if i!=bestIndex] for x in sub1X] 
            #对Lable求子集，得到subLable
            subLable=[lable[i] for i in range(len(lable)) if i!=bestIndex]
            #迭代
            branch[v]=fit(subX,subLable)  
    return tree

#测试前200行数据进行决策树
#X= np.array(X_train) 数据量太大无法运行
X= np.array(X_train[0:200])
lable =np.array(a)
mytree=fit(X,lable)
print(mytree)


# In[7]:


import matplotlib.pyplot as plt
 
#这里是对绘制是图形属性的一些定义，可以不用管，主要是后面的算法
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
 
#这是递归计算树的叶子节点个数，比较简单
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs
#这是递归计算树的深度，比较简单
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth
#这个是用来一注释形式绘制节点和箭头线，可以不用管
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
#这个是用来绘制线上的标注，简单
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
#重点，递归，决定整个树图的绘制，难（自己认为）
def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict
 
#这个是真正的绘制，上边是逻辑的绘制
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)    #no ticks
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
 
#这个是用来创建数据集即决策树
def retrieveTree(i):
    listOfTrees =[mytree]
    return listOfTrees[i]
 
createPlot(retrieveTree(0))


# In[ ]:




