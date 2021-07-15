import pandas as pd
import os
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


time_step = 45
features = 152
Aupath='E:/Desktop/dataset/一个/'
# Aupath='E:/Desktop/dataset/9stackingdata/'
fetchpath=Aupath+"fetch/"
swpath=Aupath+"pick/"
pickpath=Aupath+'sw/'
turnpath=Aupath+'turn/'
skylightpath=Aupath+'skylight/'
yawnpath=Aupath+'yawn/'
nodpath=Aupath+'nod/'
filespath=[]
filespath.append(fetchpath)
filespath.append(swpath)
filespath.append(pickpath)
filespath.append(turnpath)
filespath.append(skylightpath)
filespath.append(yawnpath)
filespath.append(nodpath)



def wholeone():
    dataX=[]
    dataY=[]
    wholefetchpath='E:/Desktop/dataset/新建文件夹/wholefetch/'
    wholeswpath='E:/Desktop/dataset/新建文件夹/wholefetch/'
    wholefetchfiles=os.listdir(wholefetchpath)
    wholeswfiles=os.listdir(wholeswpath)
    for file in wholefetchfiles:
        filepath = wholefetchpath + file
        if not os.path.isdir(file):
            tmp = pd.read_csv(filepath, header=None).values
            dataX.append(tmp)
            dataY.append([1.0, 0.0])
    for file in wholeswfiles:
        filepath = wholeswpath + file
        if not os.path.isdir(file):
            tmp = pd.read_csv(filepath, header=None).values
            dataX.append(tmp)
            dataY.append([0.0, 1.0])
    return dataX,dataY

def get4data(filepath1,filepath2,filepath3,filepath4):
    files1 = os.listdir(filepath1)
    files2 = os.listdir(filepath2)
    files3 = os.listdir(filepath3)
    files4 = os.listdir(filepath4)
    dataY=[]
    dataX=[]
    for file in files1:
        filepath=filepath1+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp)
            dataY.append([1.0,0.0,0.0,0.0])
    for file in files2:
        filepath = filepath2 + file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp)
            dataY.append([0.0,1.0,0.0,0.0])
    for file in files3:
        filepath = filepath3 + file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp)
            dataY.append([0.0,0.0,1.0,0.0])
    for file in files4:
        filepath = filepath4 + file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp)
            dataY.append([0.0,0.0,0.0,1.0])
    return dataX,dataY

#fetch0,pick1,sw2,turn3,skylight4,yawn5,nod6
#0123\2345\0124\0135\
dataX,dataY=get4data(filespath[0],filespath[1],filespath[2],filespath[3])
dataX=np.array(dataX)
dataY=np.array(dataY)

dataX=keras.preprocessing.sequence.pad_sequences(dataX,maxlen=time_step,padding='post')
dataX=np.reshape(dataX,(dataX.shape[0],time_step,features))

np.save('dataset/one_au_train_dataX.npy',dataX)
np.save('dataset/one_au_train_dataY.npy',dataY)

# np.save('dataset/one0123dataX.npy',dataX)
# np.save('dataset/one0123dataY.npy',dataY)
# np.save('dataset/AutestdataX.npy',dataX)
# np.save('dataset/AutestdataY.npy',dataY)
# np.save('dataset/merge400dataX.npy',dataX)
# np.save('dataset/merge400dataY.npy',dataY)
