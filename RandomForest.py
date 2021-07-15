
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import roc_curve, auc, classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import init_args
import process_emg
import tensorflow as tf

init=init_args.init_args()

args=init.getargs()

features=args[0]
time_step=int(args[1]/features)
print('arg:',args[0],args[1],time_step)
dataX,dataY=process_emg.get4dataall(process_emg.filespath[0],process_emg.filespath[1],process_emg.filespath[2],process_emg.filespath[3])
dataY=np.array(dataY)
dataX=np.array(dataX)
dataX=tf.keras.preprocessing.sequence.pad_sequences(dataX,maxlen=args[1],padding='post')
dataX=np.reshape(dataX,(dataX.shape[0],-1))


trainX,testX,trainY,testY=train_test_split(dataX,dataY, test_size=0.2, random_state=0)

genpath='E:/Desktop/dataset/EMG/validset/'
validX,validY=process_emg.getexpdata(genpath,features,args[1])

validX=np.reshape(validX,(validX.shape[0],-1))
print('')
def rf_model(k):
    return RandomForestClassifier(n_estimators=k, criterion="gini")


def svc_model(model):
    model.fit(trainX, trainY)
    acu_train = model.score(trainX, trainY)
    acu_test = model.score(testX, testY)
    y_pred = model.predict(testX)
    recall = recall_score(testY, y_pred, average="macro")

    y_score=model.predict(validX)
    y_ture=validY
    label = ['fetch', 'pick', 'sw', 'turn']

    ture = []
    pre = []
    for i in range(len(y_score)):
        lture = label[np.argmax(y_ture[i])]
        lpre = label[np.argmax(y_score[i])]
        ture.append(lture)
        pre.append(lpre)
    t = classification_report(ture, pre, target_names=['fetch', 'pick', 'sw', 'turn'])
    print(t)
    ta = classification_report(ture, pre, target_names=['fetch', 'pick', 'sw', 'turn'], output_dict=True)
    print(ta)
    tmp = np.zeros((4, 2))
    f2 = np.zeros((4))
    tmp[0][0] = ta['fetch']['precision']
    tmp[0][1] = ta['fetch']['recall']
    tmp[1][0] = ta['pick']['precision']
    tmp[1][1] = ta['pick']['recall']
    tmp[2][0] = ta['sw']['precision']
    tmp[2][1] = ta['sw']['recall']
    tmp[3][0] = ta['turn']['precision']
    tmp[3][1] = ta['turn']['recall']
    sum = 0

    for i in range(tmp.shape[0]):
        f2[i] = 5 * tmp[i][0] * tmp[i][1] / (4 * tmp[i][0] + tmp[i][1])
        sum = sum + f2[i]
    print(f2)
    print(sum / 4)
    print('above f2:', f2, sum / 4)


    return acu_train, acu_test, recall


def run_rf(kmax):
    result = {
        "k": [],
        "acu_train": [],
        "acu_test": [],
        "recall": []
    }
    for i in range(1, kmax + 1):
        acu_train, acu_test, recall = svc_model(rf_model(i))
        result["k"].append(i)
        result["acu_train"].append(acu_train)
        result["acu_test"].append(acu_test)
        result["recall"].append(recall)
    return pd.DataFrame(result)


df = run_rf(20)
df["acu_test"].plot()

plt.xlim(1, 100)
plt.ylim(0.5, 0.9)
plt.show()
print('testacc：',df['acu_test'])
print('recall:',df['recall'])
print('trainacc:',df['acu_train'])