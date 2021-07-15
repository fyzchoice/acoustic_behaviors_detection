import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import process_emg
from tensorflow.keras import regularizers
import tensorflow.keras as ks
import init_args
import EMG_experiments
import pandas as pd
import seaborn as sn
from tensorflow.keras.layers import Input,Dense,GRU,BatchNormalization,Masking
from sklearn.metrics import classification_report,confusion_matrix

# set GPU
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')


init=init_args.init_args()
args=init.getargs()
features=args[0]
timestep=int(args[1]/features)


#原文件读取
# print('arg:',args[0],args[1],time_step)
# dataX,dataY=process_emg.get4dataall(process_emg.filespath[0],process_emg.filespath[1],process_emg.filespath[2],process_emg.filespath[3])
# dataY=np.array(dataY)
# dataX=np.array(dataX)
# dataX=tf.keras.preprocessing.sequence.pad_sequences(dataX,maxlen=args[1],padding='post')
# dataX=np.reshape(dataX,(dataX.shape[0],-1,args[0]))

#读取npy
dataX=np.load('emg/experiment/emgtrainset3_X.npy',allow_pickle=True)
dataY=np.load('emg/experiment/emgtrainset3_Y.npy',allow_pickle=True)
dataX = tf.keras.preprocessing.sequence.pad_sequences(dataX, maxlen=args[1], padding='post')
dataX = np.reshape(dataX, (dataX.shape[0], -1, features))
trainX,testX,trainY,testY=train_test_split(dataX,dataY, test_size=0.20, random_state=1)


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


input=Input(shape=(timestep,features))
masking=Masking(mask_value=0.0,input_shape=(timestep,features))(input)
bn1=BatchNormalization()(masking)
gru1=GRU(units=128, activation='tanh',
                              kernel_regularizer=ks.regularizers.l2(0.01),
                              return_sequences=True,input_shape=(timestep,features),return_state=True,dropout=0.2)(bn1)
# bn=BatchNormalization()(gru1)
gru2=GRU(units=128,activation='tanh',kernel_regularizer=ks.regularizers.l2(0.001),dropout=0.2)(gru1)
bn2=BatchNormalization()(gru2)
dense1=Dense(units=4,activation='softmax')(bn2)
model=ks.Model(inputs=input,outputs=dense1)
model.compile(metrics=['accuracy'], loss=ks.losses.categorical_crossentropy, optimizer=ks.optimizers.Adam())

model.summary()

history=model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=15,epochs=30, verbose=1).history
model.save('Models/Emgfunctionmodel.h5')
loss,acc=model.evaluate(testX,testY)

print('loss:',loss,'acc:',acc)

EMG_experiments.sklmetric(model)
print('valid')



