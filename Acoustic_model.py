import os

from sklearn.metrics import classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras import regularizers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras


# set GPU
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

time_step = 45
features = 152


def load_data(xname,yname):
    dataX = np.load(xname)
    dataY = np.load(yname)
    return dataX,dataY

xname='dataset/allaudataX.npy'
yname='dataset/allaudataY.npy'
# xname='dataset/one0123_acoustic_dataX92.npy'
# yname='dataset/one0123_acoustic_dataY92.npy'

dataX,dataY=load_data(xname,yname)




trainX,testX,trainY,testY = train_test_split(dataX,dataY, test_size=0.3, random_state=1)


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# create and fit the LSTM network
model = tf.keras.Sequential()
#填充0后对cnn没什么影响，但是对Rnn有影响，masking层屏蔽mask_value=0.0的数据
model.add(tf.keras.layers.Masking(mask_value=0.0,input_shape=(time_step,features)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.GRU(units=128, activation='tanh',
                              kernel_regularizer=regularizers.l2(0.01),
                              return_sequences=True,input_shape=(time_step,features)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.GRU(128,activation='tanh',kernel_regularizer=regularizers.l2(0.001),
                              # return_sequences=True
                              ))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(4,activation='softmax'))
# model.add(tf.keras.layers.concatenate)
#model.compile(optimizer='adam', loss='mse')binary_crossentropy、mean_squared_error///adam
model.compile(metrics=['accuracy'], loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam())

model.summary()

history=model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=20,epochs=50, verbose=1).history
loss,acc=model.evaluate(testX,testY)
print('loss,acc:',loss,acc)
model.save("Models/aumodel.h5")




print('==============================================================')
print('==============================================================')
print('==============================================================')

tX=np.load('dataset/one_au_test_dataX.npy')
tY=np.load('dataset/one_au_test_dataY.npy')

prediction=[]
ground_truth=[]
pre=model.predict(tX)

label=['fetch', 'pick', 'sw', 'turn']
for i in range(len(pre)):
    lture = label[np.argmax(tY[i])]
    lpre = label[np.argmax(pre[i])]
    ground_truth.append(lture)
    prediction.append(lpre)
table = classification_report(prediction, ground_truth, target_names=['fetch', 'pick', 'sw', 'turn'])
print(table)
print('valid')

tmp = np.zeros((4, 2))
f2 = np.zeros((4))
tmp[0][0] = table['fetch']['precision']
tmp[0][1] = table['fetch']['recall']
tmp[1][0] = table['pick']['precision']
tmp[1][1] = table['pick']['recall']
tmp[2][0] = table['sw']['precision']
tmp[2][1] = table['sw']['recall']
tmp[3][0] = table['turn']['precision']
tmp[3][1] = table['turn']['recall']
sum = 0
for i in range(tmp.shape[0]):
    f2[i] = 5 * tmp[i][0] * tmp[i][1] / (4 * tmp[i][0] + tmp[i][1])
    sum = sum + f2[i]
avg = sum / 4
print('f2:',f2,avg)



# plt.figure(1)
# plt.plot(history['loss'], linewidth=2, label='Train')
# plt.plot(history['val_loss'], linewidth=2, label='Test')
# plt.legend(loc='upper right')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# # plt.show()
# #plt.ylim(ymin=0.70,ymax=1)
# plt.figure(2)
# plt.plot(history['accuracy'], linewidth=2, label='Trainacc')
# plt.plot(history['val_accuracy'], linewidth=2, label='Testacc')
# plt.legend(loc='upper right')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()

