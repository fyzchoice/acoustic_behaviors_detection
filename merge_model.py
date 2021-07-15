import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# set GPU
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')


#load data
def load_Audata(xname, yname):
    dataX = np.load(xname)
    dataY = np.load(yname)
    return dataX,dataY
def load_Emgdata(xname,yname):
    dataX = np.load(xname)
    dataY = np.load(yname)
    return dataX, dataY

Au_dataX, Au_dataY = load_Audata('dataset/AutestdataX.npy', 'dataset/AutestdataY.npy')
Emg_dataX, Emg_dataY = load_Audata('emg/EmgtestdataX.npy', 'emg/EmgtestdataY.npy')

Au_trainX,Au_trainY=load_Audata('dataset/austackingdataX.npy','dataset/austackingdataY.npy')
Emg_trainX,Emg_trainY=load_Audata('emg/emgstackingdataX.npy','emg/emgstackingdataY.npy')

print('')
def merge_model():
    model_1 = tf.keras.models.load_model('Models/lstm-modelall.h5')
    model_2 = tf.keras.models.load_model('Models/ConcatModels/emg_model.h5')
    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    r1 =model_1.output
    r2 =model_2.output
    x = tf.keras.layers.Concatenate(axis= 1)([r1, r2])
    model =tf.keras.Model(inputs=[inp1,inp2] ,outputs=x)
    return model

#修改模型，网络层冻结
def modify():
    origin_model=merge_model()
    for layer in origin_model.layers:
        layer.trainable=False#原来的网络层不训练

    inp=origin_model.input
    x=origin_model.output
    den=tf.keras.layers.Dense(4,activation='softmax',name='finaldense',use_bias=False)(x)
    model=tf.keras.Model(inputs=inp,outputs=den)
    return model

# merged_model=merge_model()
# pre=merged_model.predict([Au_dataX,Emg_dataX])
# print(pre)
fmodel=modify()
fmodel.summary()
fmodel.compile(metrics=['accuracy'], loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam())

history=fmodel.fit([Au_trainX,Emg_trainX], Au_trainY,validation_data=([Au_dataX,Emg_dataX],Au_dataY),batch_size=5,epochs=25, verbose=1).history
fmodel.save('Models/fmodel.h5')
loss,acc=fmodel.evaluate([Au_dataX,Emg_dataX],Emg_dataY)
print('验证集loss:',loss,'\nacc:',acc)
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
