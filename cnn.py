import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#将那些用matplotlib绘制的图显示在页面里而不是弹出一个窗口


np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # 转换成 one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Load the data
train = pd.read_csv(r'H:\model\train.csv')
test = pd.read_csv(r'H:\model\test.csv')

X_train = train.values[:,1:]
Y_train = train.values[:,0]
test=test.values

# Normalization
X_train = X_train / 255.0
test = test / 255.0
#将数组维度变成（28，28，1）
X_train = X_train.reshape(-1,28,28,1)
test = test.reshape(-1,28,28,1)

# 将标签编码为one-hot编码
# 如：2 -> [0,0,1,0,0,0,0,0,0,0]
# 7 -> [0,0,0,0,0,0,0,1,0,0]
Y_train = to_categorical(Y_train, num_classes = 10)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

# 定义cnn模型
# cnn结构为[[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# 优化算法选择RMSprop
# 损失函数选择categorical_crossentropy，亦称作多类的对数损失
# 性能评估方法选择准确率
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 30
batch_size = 86

# 设置学习率退火器
# *当评价指标不在提升时，减少学习率
# *监测量是val_acc，当3个epoch过去而模型性能不提升时，学习率减少的动作会触发
# *factor：每次减少学习率的因子，学习率将以lr = lr×factor的形式被减少
# *min_lr：学习率的下限
# *回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
# 设置图片生成器
# 用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。
datagen = ImageDataGenerator(
        featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差
        zca_whitening=False,  # 对输入数据施加ZCA白化
        rotation_range=10,  # 数据增强时图片随机转动的角度
        zoom_range = 0.1, # 随机缩放的幅度
        width_shift_range=0.1,  # 图片宽度的某个比例，数据增强时图片水平偏移的幅度
        height_shift_range=0.1,  # 图片高度的某个比例，数据增强时图片竖直偏移的幅度
        horizontal_flip=False,  # 进行随机水平翻转
        vertical_flip=False)  # 进行随机竖直翻转

# 计算依赖于数据的变换所需要的统计信息(均值方差等
datagen.fit(X_train)
# 训练模型
# fit_generator：利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率
# datagen.flow（）：生成器函数，接收numpy数组和标签为参数,生成经过数据增强或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
# callbacks=[learning_rate_reduction]：回调函数，这个list中的回调函数将会在训练过程中的适当时机被调用
import datetime
starttime = datetime.datetime.now()

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

endtime = datetime.datetime.now()

print ((endtime - starttime).seconds)
