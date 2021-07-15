import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt


# def load_Audata(xname, yname):
#     dataX = np.load(xname)
#     dataY = np.load(yname)
#     return dataX,dataY
# def load_Emgdata(xname,yname):
#     dataX = np.load(xname)
#     dataY = np.load(yname)
#     return dataX, dataY
#
#
# Au_dataX, Au_dataY = load_Audata('dataset/AutestdataX.npy', 'dataset/AutestdataY.npy')
# Emg_dataX, Emg_dataY = load_Audata('emg/EmgtestdataX.npy', 'emg/EmgtestdataY.npy')
# amodel=tf.keras.models.load_model('Models/ConcatModels/四组No497-0.5/lstm-modelNo493%.h5')
# emodel=tf.keras.models.load_model('Models/ConcatModels/四组No497-0.5/emg_model.h5')
#
# audioloss,audioacc=amodel.evaluate(Au_dataX,Au_dataY)
# emgloss,emgacc=emodel.evaluate(Emg_dataX,Emg_dataY)
#
# print('audio:',audioloss,',',audioacc)
# print('emg:',emgloss,',',emgacc)


# fmodel=tf.keras.models.load_model('Models/fmodel.h5')
#
# fmodel.summary()
#
# w=fmodel.get_layer('finaldense').get_weights()
# print(w)

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *


#生成器测试
# npar=np.random.uniform(-1, 1, [1,100])
# generator=tf.keras.models.load_model('Models/generator.h5')
# generator.summary()
#
#
# line=generator.predict(npar)[0].reshape([8*40])
#
# plt.plot(line)
# plt.show()
