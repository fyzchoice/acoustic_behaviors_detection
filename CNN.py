import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import process_emg
from tensorflow.keras import regularizers
import init_args
import EMG_experiments
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report,confusion_matrix

matplotlib.use('Agg')

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
init=init_args.init_args()
args=init.getargs()
features=args[0]
time_step=int(args[1]/features)


# dataX=np.load('emg/emg_320_80len_X.npy')
# dataY=np.load('emg/emg_320_80len_Y.npy')

# d1X=np.load('emg/experiment/r1X.npy',allow_pickle=True)
# d1Y=np.load('emg/experiment/r1Y.npy',allow_pickle=True)
# d2X=np.load('emg/experiment/emgtrainset3_X.npy',allow_pickle=True)
# d2Y=np.load('emg/experiment/emgtrainset3_Y.npy',allow_pickle=True)
#
# d2X = tf.keras.preprocessing.sequence.pad_sequences(d2X, maxlen=args[1], padding='post')
# d2X = np.reshape(d2X, (d2X.shape[0], -1, features))
#
# dataX=np.concatenate((d1X,d2X),axis=0)
# dataY=np.concatenate((d1Y,d2Y),axis=0)




print('arg:',args[0],args[1],time_step)
dataX,dataY=process_emg.get4dataall(process_emg.filespath[0],process_emg.filespath[1],process_emg.filespath[2],process_emg.filespath[3])
dataY=np.array(dataY)
dataX=np.array(dataX)
dataX=tf.keras.preprocessing.sequence.pad_sequences(dataX,maxlen=args[1],padding='post')
dataX=np.reshape(dataX,(dataX.shape[0],-1,args[0],1))


trainX,testX,trainY,testY=train_test_split(dataX,dataY, test_size=0.2, random_state=1)


print('数据集大小：',len(dataX))
print('训练集大小：',len(trainX))

model=tf.keras.Sequential()
model.add(tf.keras.layers.BatchNormalization(input_shape=(time_step,features,1)))
model.add(tf.keras.layers.Conv2D(filters=5,kernel_size=(2,10),activation='relu',input_shape=(time_step,features,1),padding='valid'))
model.add(tf.keras.layers.BatchNormalization(input_shape=(time_step,features),name='BN1'))
model.add(tf.keras.layers.Conv2D(filters=8,kernel_size=(4,10),activation='relu',padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4,activation='softmax',name='D1'))
model.compile(metrics=['accuracy'], loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam())
model.summary()
print('train:')
history=model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=25,epochs=50, verbose=1).history

model.save('cnnmodel.h5')
print('validtest1:\n')
label = ['fetch', 'pick', 'sw', 'turn']
y_ture = testY
y_pre = model.predict(testX)
ture = []
pre = []
for i in range(len(y_pre)):
    lture = label[np.argmax(y_ture[i])]
    lpre = label[np.argmax(y_pre[i])]
    ture.append(lture)
    pre.append(lpre)
table = classification_report(ture, pre, target_names=['fetch', 'pick', 'sw', 'turn'])
print(table)
t = classification_report(ture, pre, target_names=['fetch', 'pick', 'sw', 'turn'], output_dict=True)
print(t)

fetchprc = t['fetch']['precision']
pickprc = t['pick']['precision']
swprc = t['sw']['precision']
turnprc = t['turn']['precision']
avg = t['macro avg']['precision']

fetchrec = t['fetch']['recall']
pickrec = t['pick']['recall']
swrec = t['sw']['recall']
turnrec = t['turn']['recall']
avgrec = t['macro avg']['recall']

fetchf1 = t['fetch']['f1-score']
pickf1 = t['pick']['f1-score']
swf1 = t['sw']['f1-score']
turnf1 = t['turn']['f1-score']
avgf1 = t['macro avg']['f1-score']

f2_score, avgf2 = EMG_experiments.F2_score(t)
print("f2-score,avg:", f2_score, avgf2)

name_list = ['fetch', 'pick', 'sw', 'turn', 'avg']
precision = np.array([fetchprc, pickprc, swprc, turnprc, avg]) * 100
recall = np.array([fetchrec, pickrec, swrec, turnrec, avgrec]) * 100
f1_score = np.array([fetchf1, pickf1, swf1, turnf1, avgf1]) * 100
# 柱状图
x = list(range(len(precision)))
total_width, n = 0.8, 5
width = total_width / n

plt.bar(x, precision, width=width, label='precision')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, recall, width=width, label='recall', tick_label=name_list)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, f1_score, width=width, label='f1-score', tick_label=name_list)
plt.ylabel('Percentage(%)')
plt.ylim((60, 100))
plt.legend()
plt.show()
# 混淆矩阵
confusion = confusion_matrix(ture, pre)
fig, ax = plt.subplots(figsize=(8, 6))
sn.heatmap(confusion, annot=True, fmt='.5g', cmap='YlGnBu', linewidths=.5,
           xticklabels=['fetch', 'pick', 'sw', 'turn'], yticklabels=['fetch', 'pick', 'sw', 'turn'])  # acent
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# plt.show()
plt.savefig('test.eps',format='eps')
EMG_experiments.drawROC(y_ture, y_pre)

print('sklmetric:')
EMG_experiments.sklmetric(model)