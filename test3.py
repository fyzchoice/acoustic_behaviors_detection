from sklearn.metrics import classification_report
import tensorflow as tf
import process_emg
import numpy as np


# set GPU
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

#
# prediction=[]
# ground_truth=[0,1,2,2]
# pre=np.random.rand(4,4)
#
# print(pre)
# for i in range(len(pre)):
#     prediction.append(np.argmax(pre[i]))
#
# print(prediction)
# print(prediction[3])
#
# table = classification_report(prediction, ground_truth, target_names=['fetch', 'pick', 'sw', 'turn'])
# print(table)
#
# vvv=np.load('dataset/one0123_acoustic_dataX92.npy')
# ttt=np.load('dataset/AutestdataX.npy')
# kkk=np.load('dataset/one_au_test_dataX.npy')
# print('')
tX=np.load('dataset/one_au_test_dataX.npy')
tY=np.load('dataset/one_au_test_dataY.npy')
model=tf.keras.models.load_model('Models/excllentmodel/onevalid96aumodel.h5')
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