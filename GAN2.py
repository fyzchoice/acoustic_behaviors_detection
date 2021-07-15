import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from IPython.display import clear_output # jupyter清屏函数
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

# (train_x, train_y), (test_x, test_y) = mnist.load_data()
# train_x = train_x.reshape([-1, 28 * 28]) / 255 # flatten后归一化

train_x=np.load('emg/onefetch_sw_pick_turn_dataX.npy')

train_x=train_x.reshape([-1,8*40])

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

### 生成器
generator = Sequential([
    LSTM(128, activation='tanh', input_shape=[1,100],return_sequences=True),
    BatchNormalization(),
    LeakyReLU(),
    LSTM(512,activation='tanh'),
    BatchNormalization(),
    LeakyReLU(),
    Dense(8 * 40, activation='sigmoid')
])

### 判别器
discriminator = Sequential([
    LSTM(512, activation='tanh', input_shape=[1,8 * 40],return_sequences=True),
    BatchNormalization(),
    LeakyReLU(),
    LSTM(128,activation='tanh'),
    BatchNormalization(),
    LeakyReLU(),
    Dense(1, activation='sigmoid')
])

g_sample_input = Input([100]) # 生成器输入
x_input = Input([8 * 40]) # 真实数据输入

### 裁剪概率到区间[1e-6, 1]内，并求其log，避免log后为inf，K.stop_gradient表示训练时不对其求梯度
#   这里也可直接写成 log_clip = Lambda(lambda x: K.log(x + 1e-3))
log_clip = Lambda(lambda x: K.log(K.clip(K.stop_gradient(x), 1e-6, 1) - K.stop_gradient(x) + x))

g = discriminator(generator(g_sample_input)) # 假数据

### 判别器loss
d_loss = (
    - log_clip(discriminator(x_input))
    - log_clip(1.0 - g)
)

fit_discriminator = Model(inputs=[x_input, g_sample_input], outputs=d_loss) # 训练discriminator所用模型
fit_discriminator.add_loss(d_loss) # 添加自定义loss

### 在调用compile之前置generator.trainable为False，调用compile后的模型训练时不更新generator的参数
generator.trainable = False
fit_discriminator.compile(optimizer=Adam(0.001))
generator.trainable = True

### 生成器loss
g_loss = (
    - log_clip(g)
)

fit_generator = Model(inputs=g_sample_input, outputs=g_loss) # 训练generator所用模型
fit_generator.add_loss(g_loss)

### 生成器训练时不更新discriminator的参数
discriminator.trainable = False
fit_generator.compile(optimizer=Adam(0.001))
discriminator.trainable = True



batch_size = 64
for i in range(20000):
    # if i % 50 == 0:
    #     clear_output()
    #     plt.plot(generator.predict(np.random.uniform(-1, 1, [1, 100]))[0].reshape([8*40]))
    #     plt.show()
    print(i)
    x = train_x[random.sample(range(len(train_x)), batch_size)] # 随机选取batch_size个真样本
    g_sample = np.random.uniform(-1, 1, [batch_size, 100]) # 生成batch_size个随机数据输入
    fit_discriminator.fit([K.constant(x), K.constant(g_sample)]) # 训练辨别器，多输入需传入一个包含多个tensor的列表，此处用K.constant代替
    fit_generator.fit(g_sample) # 训练生成器


generator.save('Models/generator.h5')

fig, axes = plt.subplots(3, 3, figsize=(3, 3))
for i in range(3):
    for j in range(3):
        axes[i, j].plot(generator.predict(np.random.uniform(-1, 1, [1, 100]))[0].reshape([8*40]))
        axes[i, j].axis(False)
plt.show()





