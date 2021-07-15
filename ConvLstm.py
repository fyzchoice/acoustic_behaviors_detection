import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from keras import regularizers
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import LSTM, GRU, Dense
import tensorflow.keras as tfk
import matplotlib.pyplot as plt

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')


timestep=45
features=152

dataX=np.load('dataset/one0123_acoustic_dataX.npy')
dataY=np.load('dataset/one0123_acoustic_dataY.npy')

trainX,testX,trainY,testY = train_test_split(dataX,dataY, test_size=0.2, random_state=0)

def getmodel():
    model=tfk.Sequential()
    model.add(tfk.layers.Masking(input_shape=(timestep,features)))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.Conv1D(64, 8, strides=2,activation='relu',padding='same'))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.LSTM(128, dropout=0.2, return_sequences=True,kernel_regularizer=regularizers.l2(0.01)))
    model.add(tfk.layers.BatchNormalization())
    model.add(tfk.layers.LSTM(128, dropout=0.2))
    model.add(tfk.layers.Dense(5, activation="softmax"))
    return model
model=getmodel()
model.compile(metrics=['accuracy'], loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam())
model.summary()

history=model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=25,epochs=50, verbose=1).history

loss,acc=model.evaluate(testX,testY)
print('loss:',loss,'acc:',acc)
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(ymin=0.70,ymax=1)
plt.show()