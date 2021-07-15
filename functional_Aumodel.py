import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from tensorflow.keras.layers import Input,Dense,GRU,BatchNormalization,Masking
from sklearn.model_selection import train_test_split

# set GPU
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')


def load_data(xname,yname):
    dataX = np.load(xname)
    dataY = np.load(yname)
    return dataX,dataY

# xname='dataset/one_au_train_dataX.npy'
# yname='dataset/one_au_train_dataY.npy'
xname='dataset/allaudataX.npy'
yname='dataset/allaudataY.npy'
dataX,dataY=load_data(xname,yname)

trainX,testX,trainY,testY = train_test_split(dataX,dataY, test_size=0.2, random_state=1)


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)



timestep=45
features=152

input=Input(shape=(timestep,features))
masking=Masking(mask_value=0.0,input_shape=(timestep,features))(input)
bn1=BatchNormalization()(input)
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

history=model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=20,epochs=50, verbose=1).history
model.save('Models/aufunctionmodel.h5')
loss,acc=model.evaluate(testX,testY)
print(xname)
print('loss:',loss,'acc:',acc)

tX=np.load('dataset/one_au_test_dataX.npy')
tY=np.load('dataset/one_au_test_dataY.npy')

lo,ac=model.evaluate(tX,tY)
print('lo,ac:',lo,ac)
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




