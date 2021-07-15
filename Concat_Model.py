import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.python.keras.constraints import MaxNorm
from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras

#args
Au_time_step = 45
Au_features = 152
Emg_time_step=8
Emg_features=40

# set GPU
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

alpha=0.50
beta=0.50
label=['fetch','pick','sw','turn']

# Au_model=tf.keras.models.load_model('Models/ConcatModels/lstm-modelNo4.h5')
Au_model=tf.keras.models.load_model('Models/lstm-modelall.h5')
Emg_model=tf.keras.models.load_model('Models/ConcatModels/emg_model.h5')



#load data
def load_Audata(xname, yname):
    dataX = np.load(xname)
    dataY = np.load(yname)
    return dataX,dataY
def load_Emgdata(xname,yname):
    dataX = np.load(xname)
    dataY = np.load(yname)
    return dataX, dataY

def evaluate_fusion(Au_model,Emg_model):
    Au_dataX, Au_dataY = load_Audata('dataset/AutestdataX.npy', 'dataset/AutestdataY.npy')
    Emg_dataX, Emg_dataY=load_Audata('emg/EmgtestdataX.npy', 'emg/EmgtestdataY.npy')
    len=Au_dataX.shape[0]
    right1=0
    right2 = 0
    right = 0
    unknown=0
    for m in range(len-1):
        pre1=Au_model.predict(Au_dataX[m].reshape(1,45,152))
        pre2=Emg_model.predict(Emg_dataX[m].reshape(1,8,40))
        real=np.argmax(Au_dataY[m])
        fpre=alpha*pre1+beta*pre2
        if(np.argmax(pre1)==real):
            right1=right1+1
        if(np.argmax(pre2)==real):
            right2=right2+1
        if(np.argmax(fpre)==real):
            right=right+1
        if(fpre.max()<0.6):
            unknown=unknown+1
    print('unk(p<0.6):',unknown)
    print('单个aduioacc:',right1/len)
    print('单个emgacc:',right2/len)
    print('acc:',right/len)


def evaluate_audio(Au_model):
    Au_dataX, Au_dataY = load_Audata('dataset/AutestdataX.npy', 'dataset/AutestdataY.npy')
    rfetch=0
    rpick=0
    rsw=0
    rturn=0
    l = int(Au_dataX.shape[0]/4)
    for index,item in enumerate(Au_dataX):
        if(index>=0 and index<l):
            pre=Au_model.predict(item.reshape(1,Au_time_step,Au_features))
            real=Au_dataY[index]
            if(np.argmax(pre)==np.argmax(real)):
                rfetch=rfetch+1
        if(index>=l and index<2*l):
            pre=Au_model.predict(item.reshape(1,Au_time_step,Au_features))
            real=Au_dataY[index]
            if(np.argmax(pre)==np.argmax(real)):
                rpick=rpick+1
        if(index>=2*l and index<3*l):
            pre=Au_model.predict(item.reshape(1,Au_time_step,Au_features))
            real=Au_dataY[index]
            if(np.argmax(pre)==np.argmax(real)):
                rsw=rsw+1
        if(index>=3*l and index<4*l):
            pre=Au_model.predict(item.reshape(1,Au_time_step,Au_features))
            real=Au_dataY[index]
            if(np.argmax(pre)==np.argmax(real)):
                rturn=rturn+1
    print('fetchacc:',rfetch/ l,'\n'
          'pickacc:', rpick / l, '\n'
          'swacc:', rsw / l, '\n'
          'turnacc:', rturn / l, '\n',
          'Acc',(rfetch+rpick+rsw+rturn)/(4*l)
          )






# evaluate_fusion(Au_model,Emg_model)
evaluate_audio(Au_model)




# def ConacatModel():
#     # create and fit the LSTM network
#     model = tf.keras.Sequential()
#     l1=tf.keras.layers.Masking(mask_value=0.0,input_shape=(Au_time_step,Au_features))
#     l2=tf.keras.layers.Masking(mask_value=0.0,input_shape=(Emg_time_step,Emg_features))
#     ConcatLayer=tf.keras.layers.concatenate([l1,l2])
#     #填充0后对cnn没什么影响，但是对Rnn有影响，masking层屏蔽mask_value=0.0的数据
#     model.add(ConcatLayer)
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.LSTM(units=128, activation='tanh',
#                                   kernel_regularizer=regularizers.l2(0.01),
#                                   return_sequences=True))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.LSTM(128,activation='tanh',kernel_regularizer=regularizers.l2(0.001),
#                                   ))
#     model.add(tf.keras.layers.Dropout(0.2))
#     # model.add(tf.keras.layers.BatchNormalization())
#     # model.add(tf.keras.layers.GRU(108,activation='tanh',kernel_regularizer=regularizers.l2(0.01)))
#     # model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.Dense(4,activation='softmax'))
#     # model.add(tf.keras.layers.concatenate)
#     #model.compile(optimizer='adam', loss='mse')binary_crossentropy、mean_squared_error///adam
#     model.compile(metrics=['accuracy'], loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam())
#     model.summary()
#     return model








