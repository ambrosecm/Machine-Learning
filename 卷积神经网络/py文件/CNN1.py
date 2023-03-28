#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入所需模块
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras import backend as K

class SimpleVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # 创建Sequential 顺序模型层状结构
        model = Sequential()
        # 输入层的图片长度、宽度、深度（深度为RGB 3层）
        inputShape = (height, width, depth)
        chanDim = -1
        
        # 通过通道维所在位置修改chanDim
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        # 添加卷积层，滤波器数量为32，卷积核大小为3*3（过滤器使用kernel_initializer参数指定的方法初始化为小的随机值）
        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        #添加激活功能层，激活函数为relu
        model.add(Activation("relu"))
        #对数据做批规范化
        model.add(BatchNormalization(axis=chanDim))
        #添加最大池化层，池化窗口大小为2*2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #dropout层减少神经元，降低过拟合
        #model.add(Dropout(0.25))

        # FC层（全连接层）
        #添加Flatten层把多维的输入一维化
        model.add(Flatten())
        #添加全连接层，神经元点个数为512
        model.add(Dense(512,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(512,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        # softmax 分类
        #添加全连接层，神经元点个数为10
        model.add(Dense(classes,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("softmax"))

        return model
    
class VGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # 创建Sequential 顺序模型层状结构
        model = Sequential()
        # 输入层的图片长度、宽度、深度（深度为RGB 3层）
        inputShape = (height, width, depth)
        chanDim = -1
        
        # 通过通道维所在位置修改chanDim
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        model.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='same'))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='same'))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(256, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(256, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(256, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='same'))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(512, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(512, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(512, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='same'))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(512, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(512, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        
        model.add(Conv2D(512, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='same'))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(classes,activation='softmax'))
        return model 
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # 创建Sequential 顺序模型层状结构
        model = Sequential()
        # 输入层的图片长度、宽度、深度（深度为RGB 3层）
        inputShape = (height, width, depth)
        chanDim = -1
        
        # 通过通道维所在位置修改chanDim
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        model.add(Conv2D(6, (5, 5),activation='sigmoid',input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Conv2D(16, (5, 5),activation='sigmoid',kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Flatten())
        model.add(Dense(120,activation='sigmoid'))
        model.add(Dense(84,activation='sigmoid'))
        model.add(Dense(classes,activation='softmax'))
        return model
class AlexNet:
    @staticmethod
    def build(width, height, depth, classes):
        # 创建Sequential 顺序模型层状结构
        model = Sequential()
        # 输入层的图片长度、宽度、深度（深度为RGB 3层）
        inputShape = (height, width, depth)
        chanDim = -1
        
        # 通过通道维所在位置修改chanDim
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        model.add(Conv2D(96, (3, 3),input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
        
        model.add(Conv2D(256, (3, 3),kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
        
        model.add(Conv2D(384, (3, 3),padding='same', activation='relu',kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Conv2D(384, (3, 3),padding='same', activation='relu',kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
        
        model.add(Flatten())
        model.add(Dense(2048,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes,activation='softmax'))
        return model


# In[2]:


import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
 
def list_images(basePath, contains=None):
    # 返回有效的图片路径数据集
    return list_files(basePath, validExts=image_types, contains=contains)
 
def list_files(basePath, validExts=None, contains=None):
    # 遍历图片数据目录，生成每张图片的路径
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # 循环遍历当前目录中的文件名
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
 
            # 通过确定.的位置，从而确定当前文件的文件扩展名
            ext = filename[filename.rfind("."):].lower()
 
            # 检查文件是否为图像，是否应进行处理
            if validExts is None or ext.endswith(validExts):
                # 构造图像路径
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


# In[3]:


# 导入所需工具包
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# 读取数据和标签
print("------开始读取数据------")
data = []
labels = []

# 得到图像数据路径
imagePaths = sorted(list(list_images('F:/img')))
# 改变随机数生成器的种子
random.seed(42)
# 将一个图像地址列表中的元素打乱
random.shuffle(imagePaths)

# 遍历读取数据
for imagePath in imagePaths:
    # 读取图像数据
    image = cv2.imread(imagePath)
    # 缩放图像为64*64
    image = cv2.resize(image, (64, 64))
    data.append(image)
    # 读取标签
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

print(data[0])

# 对图像数据做scale操作
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print(data[0])

# 数据集切分,验证集0.3
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.3, random_state=42)

# 转换标签为one-hot 编码格式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print(trainY[0])

# 数据增强处理
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


# In[5]:


# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 50
BS = 32

# 建立卷积神经网络
model = SimpleVGGNet.build(width=64, height=64, depth=3,classes=len(lb.classes_))

# 损失函数，编译模型
print("------准备训练网络------")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# 训练网络模型
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)
"""
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
"""



# In[4]:


# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 50
BS = 32

# 建立卷积神经网络
model_vggnet = VGGNet.build(width=64, height=64, depth=3,classes=len(lb.classes_))

# 损失函数，编译模型
print("------准备训练网络------")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model_vggnet.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# 训练网络模型
H_vggnet = model_vggnet.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)
"""
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
"""



# In[6]:


# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 50
BS = 32

# 建立卷积神经网络
model_lenet = LeNet.build(width=64, height=64, depth=3,classes=len(lb.classes_))

# 损失函数，编译模型
print("------准备训练网络------")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model_lenet.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# 训练网络模型
H_lenet = model_lenet.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)
"""
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
"""



# In[22]:


# 设置初始化超参数
INIT_LR = 0.01
EPOCHS = 50
BS = 32

# 建立卷积神经网络
model_alexnet = AlexNet.build(width=64, height=64, depth=3,classes=len(lb.classes_))

# 损失函数，编译模型
print("------准备训练网络------")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model_alexnet.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# 训练网络模型
H_alexnet = model_alexnet.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)
"""
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
"""


# In[8]:


# 测试
print("------测试网络------")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# 绘制结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('F://output/cnn_plot.png')

# 保存模型
print("------正在保存模型------")
model.save('F://output/cnn.model')
f = open('F://output/cnn_lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


# In[9]:


# 测试
print("------测试网络------")
predictions = model_vggnet.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# 绘制结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H_vggnet.history["loss"], label="train_loss")
plt.plot(N, H_vggnet.history["val_loss"], label="val_loss")
plt.plot(N, H_vggnet.history["accuracy"], label="train_acc")
plt.plot(N, H_vggnet.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('F://output/cnn_plot_vggnet.png')

# 保存模型
print("------正在保存模型------")
model.save('F://output/cnn_vggnet.model')
f = open('F://output/cnn_lb_vggnet.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


# In[10]:


# 测试
print("------测试网络------")
predictions = model_lenet.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# 绘制结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H_lenet.history["loss"], label="train_loss")
plt.plot(N, H_lenet.history["val_loss"], label="val_loss")
plt.plot(N, H_lenet.history["accuracy"], label="train_acc")
plt.plot(N, H_lenet.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('F://output/cnn_plot_lenet.png')

# 保存模型
print("------正在保存模型------")
model.save('F://output/cnn_lenet.model')
f = open('F://output/cnn_lb_lenet.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


# In[23]:


# 测试
print("------测试网络------")
predictions = model_alexnet.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# 绘制结果曲线
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H_alexnet.history["loss"], label="train_loss")
plt.plot(N, H_alexnet.history["val_loss"], label="val_loss")
plt.plot(N, H_alexnet.history["accuracy"], label="train_acc")
plt.plot(N, H_alexnet.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('F://output/cnn_plot_alexnet.png')

# 保存模型
print("------正在保存模型------")
model.save('F://output/cnn_alexnet.model')
f = open('F://output/cnn_lb_alexnet.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


# In[21]:


# 导入所需工具包
from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2


# 加载测试数据并进行相同预处理操作
image = cv2.imread('F:/imgtest/2/7.png')
output = image.copy()
image = cv2.resize(image, (64, 64))

# scale图像数据
image = image.astype("float") / 255.0

# 对图像进行拉平操作
image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

# 读取模型和标签
print("------读取模型和标签------")
model = load_model('F://output/cnn.model')
lb = pickle.loads(open('F://output/cnn_lb.pickle', "rb").read())

# 预测
preds = model.predict(image)

# 得到预测结果以及其对应的标签
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# 在图像中把结果画出来
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255), 1)

# 绘图
cv2.imshow("Image", output)
cv2.waitKey(0)


# In[4]:





# In[ ]:




