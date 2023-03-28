#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[4]:


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
print(sms_data)


# In[5]:


#按0.7：0.3比例分为训练集和测试集
dataset_size=len(sms_data)
trainset_size=int(round(dataset_size*0.7))
print('dataset_size:',dataset_size,'trainset_size:',trainset_size)


# In[6]:


# 将数据向量化
x_train=np.array([''.join(el) for el in sms_data[0:trainset_size]])
y_train=np.array(sms_label[0:trainset_size])

x_test=np.array(sms_data[trainset_size+1:dataset_size])
y_test=np.array(sms_label[trainset_size+1:dataset_size])

# 使用词袋模型
vectorizer=TfidfVectorizer(min_df=2,ngram_range=(1,2),stop_words='english',strip_accents='unicode',norm='l2')
# 把文本全部转换为小写，然后将文本词块化。主要是分词，分标点
X_train=vectorizer.fit_transform(x_train)
X_test=vectorizer.transform(x_test)


# In[7]:


# 决策树
#ID3算法使用的是entropy，CART算法使用的则是gini
tree1=tree.DecisionTreeClassifier(criterion='gini')
clf=tree1.fit(X_train.toarray(),y_train)
y_tree_pred=clf.predict(X_test.toarray())

print('tree_confusion_matrix:')
cm=confusion_matrix(y_test,y_tree_pred)
print(cm)

print('tree_classification_report:')
print(classification_report(y_test,y_tree_pred))


# In[8]:


y_pred = clf.predict(X_test.toarray())
y_pred_proba = clf.predict_proba(X_test.toarray())[:,1]

#准确率
acc = accuracy_score(y_test,y_pred)
#roc 曲线
fpr,tpr,threshold = roc_curve(y_test,y_pred_proba,pos_label='spam')
auc = roc_auc_score(y_test,y_pred_proba)  
#可视化
plt.figure()
#plt.rcParams["font"]
plt.plot(fpr,tpr,"ro-",label="roc curve")
plt.title("auc %.1f,acc %f.1"%(auc,acc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.legend(loc="best")


# In[9]:


os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'  #注意修改你的路径

clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=10).fit(X_train.toarray(),y_train)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("spam1.pdf") 


# In[11]:


test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i+1
                                      ,criterion="entropy"
                                      ,random_state=30
                                      ,splitter="random"
                                     )
    clf = clf.fit(X_train.toarray(),y_train)
    score = clf.score(X_test.toarray(), y_test)
    test.append(score)
plt.plot(range(1,11),test,color="red",label="max_depth")
plt.legend()
plt.show()


# In[12]:


clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=9,random_state=0).fit(X_train.toarray(),y_train)
y_tree_pred=clf.predict(X_test.toarray())

print('tree_confusion_matrix:')
cm=confusion_matrix(y_test,y_tree_pred)
print(cm)

print('tree_classification_report:')
print(classification_report(y_test,y_tree_pred))


# In[ ]:




