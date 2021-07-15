import pandas as pd
import numpy as np
import os
import tensorflow as tf
import init_args
from sklearn.model_selection import train_test_split
import csv

init=init_args.init_args()
args=init.getargs()
Emg_features=args[0]
Emg_time_step=int(args[1]/Emg_features)

emgpath="E:/Desktop/dataset/EMG/trainset3 overperformence/"
# emgpath="E:/Desktop/dataset/一个/testdata/emg/"
# emgpath="E:/Desktop/dataset/EMG/trainset/"
# emgpath='E:/Desktop/dataset/9stackingdata/emg/'
# emgpath="E:/Desktop/dataset/EMG/"
fetchpath=emgpath+'fetch/'
swpath=emgpath+'pick/'
pickpath=emgpath+'sw/'
turnpath=emgpath+'turn/'
skylightpath=emgpath+'sky/'
yawnpath=emgpath+'yawn/'

filespath=[]
filespath.append(fetchpath)
filespath.append(swpath)
filespath.append(pickpath)
filespath.append(turnpath)
filespath.append(skylightpath)
filespath.append(yawnpath)

def get6data(filespath):
    files1=os.listdir(filespath[0])
    files2 = os.listdir(filespath[1])
    files3 = os.listdir(filespath[2])
    files4 = os.listdir(filespath[3])
    files5=os.listdir(filespath[4])
    files6 = os.listdir(filespath[5])
    dataY=[]
    dataX=[]
    for file in files1:
        filepath=filespath[0]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([1.0,0.0,0.0,0.0,0.0,0.0,0.0])
    for file in files2:
        filepath=filespath[1]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,1.0,0.0,0.0,0.0,0.0,0.0])
    for file in files3:
        filepath=filespath[2]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,1.0,0.0,0.0,0.0,0.0])
    for file in files4:
        filepath=filespath[3]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,0.0,1.0,0.0,0.0,0.0])
    for file in files5:
        filepath=filespath[4]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,0.0,0.0,1.0,0.0,0.0])
    for file in files6:
        filepath=filespath[5]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,0.0,0.0,0.0,1.0,0.0])
    return dataX,dataY

def get6data_msq(filespath):
    files1=os.listdir(filespath[0])
    files2 = os.listdir(filespath[1])
    files3 = os.listdir(filespath[2])
    files4 = os.listdir(filespath[3])
    files5=os.listdir(filespath[4])
    files6 = os.listdir(filespath[5])
    dataY=[]
    dataX=[]
    for file in files1:
        filepath=filespath[0]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([1.0,0.0,0.0,0.0,0.0,0.0,0.0])
    for file in files2:
        filepath=filespath[1]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,1.0,0.0,0.0,0.0,0.0,0.0])
    for file in files3:
        filepath=filespath[2]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,1.0,0.0,0.0,0.0,0.0])
    for file in files4:
        filepath=filespath[3]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,0.0,1.0,0.0,0.0,0.0])
    for file in files5:
        filepath=filespath[4]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,0.0,0.0,1.0,0.0,0.0])
    for file in files6:
        filepath=filespath[5]+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,0.0,0.0,0.0,1.0,0.0])
    return dataX,dataY





def get4data(filepath1,filepath2,filepath3,filepath4):
    files1=os.listdir(filepath1)
    files2 = os.listdir(filepath2)
    files3 = os.listdir(filepath3)
    files4 = os.listdir(filepath4)
    dataY=[]
    dataX=[]
    i=0
    j=0
    k=0
    l=0
    for file in files1:
        if(i%3==0):
            i=i+1
            continue
        else:
            filepath=filepath1+file
            if not os.path.isdir(file):
                tmp=pd.read_csv(filepath,header=None).values
                dataX.append(tmp[:,1])
                dataY.append([1.0,0.0,0.0,0.0])
        i=i+1
    for file in files2:
        if(k%3==0):
            k=k+1
            continue
        else:
            filepath = filepath2 + file
            if not os.path.isdir(file):
                tmp=pd.read_csv(filepath,header=None).values
                dataX.append(tmp[:,1])
                dataY.append([0.0,1.0,0.0,0.0])
        k=k+1
    for file in files3:
        if(j%3==0):
            j=j+1
            continue
        else:
            filepath = filepath3 + file
            if not os.path.isdir(file):
                tmp=pd.read_csv(filepath,header=None).values
                dataX.append(tmp[:,1])
                dataY.append([0.0,0.0,1.0,0.0])
        j=j+1
    for file in files4:
        if(l%3==0):
            l=l+1
            continue
        else:
            filepath = filepath4 + file
            if not os.path.isdir(file):
                tmp=pd.read_csv(filepath,header=None).values
                dataX.append(tmp[:,1])
                dataY.append([0.0,0.0,0.0,1.0])
        l=l+1
    return dataX,dataY


def get4dataall(filepath1,filepath2,filepath3,filepath4):
    files1=os.listdir(filepath1)
    files2 = os.listdir(filepath2)
    files3 = os.listdir(filepath3)
    files4 = os.listdir(filepath4)
    dataY=[]
    dataX=[]

    for file in files1:
        filepath=filepath1+file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([1.0,0.0,0.0,0.0])

    for file in files2:
        filepath = filepath2 + file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,1.0,0.0,0.0])

    for file in files3:
        filepath = filepath3 + file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,1.0,0.0])
    for file in files4:
        filepath = filepath4 + file
        if not os.path.isdir(file):
            tmp=pd.read_csv(filepath,header=None).values
            dataX.append(tmp[:,1])
            dataY.append([0.0,0.0,0.0,1.0])

    for index, m in enumerate(dataX):
        max=m.max()
        min=m.min()
        avg = m.mean()
        dataX[index] = np.square(dataX[index]-avg)/max

    return dataX,dataY




dataX,dataY=get4data(filespath[0],filespath[1],filespath[2],filespath[3])
# dataX,dataY=get6data(filespath=filespath)

dataY=np.array(dataY)
dataX=np.array(dataX)

# for index, m in enumerate(dataX):
#     # max=m.max()
#     avg = m.mean()
#     dataX[index] = np.sqrt(dataX[index] - avg)


# features=80
# dataX=tf.keras.preprocessing.sequence.pad_sequences(dataX,maxlen=320,padding='post')
# dataX=np.reshape(dataX,(dataX.shape[0],-1,features))


def getexpdata(emgpath,features,pad_len):
    emgpath=emgpath
    fetchpath = emgpath + 'fetch/'
    swpath = emgpath + 'pick/'
    pickpath = emgpath + 'sw/'
    turnpath = emgpath + 'turn/'
    filespath = []
    filespath.append(fetchpath)
    filespath.append(swpath)
    filespath.append(pickpath)
    filespath.append(turnpath)
    dataX,dataY=get4dataall(filespath[0],filespath[1],filespath[2],filespath[3])
    dataY = np.array(dataY)
    dataX = np.array(dataX)
    dataX = tf.keras.preprocessing.sequence.pad_sequences(dataX, maxlen=pad_len, padding='post')
    dataX = np.reshape(dataX, (dataX.shape[0], -1, features))
    return dataX,dataY

# dX,dY=getexpdata('E:/Desktop/dataset/EMG/r3/', features=Emg_features,pad_len=args[1])
# dX=np.array(dX)
# dY=np.array(dY)
# # dX=tf.keras.preprocessing.sequence.pad_sequences(dX,maxlen=args[1],padding='post')
# # dX=np.reshape(dX,(dX.shape[0],-1,args[0]))
#
# np.save('emg/experiment/r3X.npy', dX)
# np.save('emg/experiment/r3Y.npy', dY)



# np.save('emg/onefetch_sw_pick_turn_dataX.npy',dataX)
# np.save('emg/onefetch_sw_pick_turn_dataY.npy',dataY)
# np.save('emg/EmgtestdataX.npy',dataX)
# np.save('emg/EmgtestdataY.npy',dataY)
# np.save('emg/emg_320_80len_X.npy',dataX)
# np.save('emg/emg_320_80len_Y.npy',dataY)
# np.save('emg/experiment/emgtrainset_X.npy', dataX)
# np.save('emg/experiment/emgtrainset_Y.npy', dataY)



print('')