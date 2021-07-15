import tensorflow as tf
import numpy as np
import pandas as pd
label=['fetch','pick','sw','turn']
import os
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')



# model=tf.keras.models.load_model('Models/lstm-model92.h5')
# model.summary()


def load_testdata(akindpath,Y):
    files = os.listdir(akindpath)
    dataX = []
    dataY=[]
    for file in files:
        filepath = akindpath + file
        if not os.path.isdir(file):
            tmp = pd.read_csv(filepath, header=None).values
            dataX.append(tmp)
            dataY.append(Y)
    return dataX,dataY

# dense_weight=model.get_layer('dense').get_weights()
# print(dense_weight)
# testdata=pd.read_csv('dataset/testfile/2pick1110_20_43_34_13.csv').values
# testdata=np.array(testdata)
# testdata=np.reshape(testdata,(1,testdata.shape[0],152))

# testX,testY=load_testdata('E:/Desktop/dataset/一个/turn/',[0.0,0.0,0.0,1.0,0.0])
#
# testX=np.array(testX)
# testY=np.array(testY)
# testX=tf.keras.preprocessing.sequence.pad_sequences(testX,maxlen=45,padding='post',value=0.0)
# # # testX=np.reshape(testX,(testX.shape[0],45,152))
# np.save('dataset/testfile/turnX.npy',testX)
# np.save('dataset/testfile/turnY.npy',testY)

# testX=np.load('dataset/testfile/pickX.npy')
# testY=np.load('dataset/testfile/pickY.npy')

# print('')
# loss,acc=model.evaluate(testX,testY)
# print('loss:',loss,'acc:',acc)

#pick97%、sw98%
#fetch97%、95%


