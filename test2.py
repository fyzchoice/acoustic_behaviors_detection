import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report,auc
import tensorflow as tf

model_path=(r'Models/emg_model.h5')
model = tf.keras.models.load_model(model_path)
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()
open('emg.lite','wb').write(tflite_model)

# groudth: ['fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn']
# predict: ['fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'pick',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'pick', 'fetch', 'fetch', 'fetch', 'turn', 'fetch', 'pick', 'fetch', 'fetch', 'fetch', 'pick',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'pick', 'fetch', 'pick', 'fetch', 'fetch', 'pick', 'turn', 'fetch', 'fetch', 'pick', 'pick',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'sw', 'pick', 'fetch', 'pick', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'pick', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'pick', 'pick', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'pick', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'pick', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'pick',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'pick',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'pick', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch',
#           'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'fetch', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'turn', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'turn', 'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick', 'sw', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'sw', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'sw', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'sw', 'sw', 'sw', 'pick', 'pick', 'pick', 'pick', 'sw', 'sw', 'sw', 'pick',
#           'pick', 'pick', 'sw', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'fetch', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'sw', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick', 'pick',
#           'pick', 'pick', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw',
#           'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'sw', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'pick', 'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'pick',
#           'turn', 'turn', 'turn', 'turn', 'pick', 'pick', 'turn', 'pick', 'turn', 'turn', 'pick', 'turn', 'turn',
#           'turn', 'turn', 'pick', 'turn', 'turn', 'pick', 'pick', 'turn', 'pick', 'pick', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'pick',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'pick', 'pick',
#           'turn', 'fetch', 'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'pick', 'turn', 'turn', 'turn', 'turn',
#           'pick', 'pick', 'pick', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'turn', 'pick', 'sw', 'pick', 'turn', 'pick', 'fetch',
#           'fetch', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'pick', 'pick', 'fetch', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'fetch', 'fetch', 'fetch', 'turn', 'fetch', 'turn', 'fetch', 'pick', 'turn',
#           'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'pick', 'pick', 'pick', 'turn', 'turn', 'turn', 'turn',
#           'pick', 'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'pick', 'pick',
#           'turn', 'fetch', 'pick', 'pick', 'pick', 'turn', 'fetch', 'turn', 'turn', 'pick', 'fetch', 'sw', 'sw', 'turn',
#           'turn', 'turn', 'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'pick', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn',
#           'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn']

#折线图
# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
# plt.figure(figsize=(10, 5))
# plt.grid(linestyle="--")  # 设置背景网格线为虚线
# ax = plt.gca()
# ax.spines['top'].set_visible(False)  # 去掉上边框
# ax.spines['right'].set_visible(False)  # 去掉右边框
#
# plt.plot(x, VGG_supervised, marker='v', color="blue", label="VGG-style Supervised Network", linewidth=1.5)
# plt.plot(x, VGG_unsupervised, marker='o', color="green", label="VGG-style Unsupervised Network", linewidth=1.5)
# plt.plot(x, ourNetwork, marker='*', color="red", label="ShuffleNet-style Network", linewidth=1.5)
#
# group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
# plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
# plt.yticks(fontsize=12, fontweight='bold')
# # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
# plt.xlabel("Performance Percentile", fontsize=13, fontweight='bold')
# plt.ylabel("4pt-Homography RMSE", fontsize=13, fontweight='bold')
# plt.xlim(0.9, 6.1)  # 设置x轴的范围
# plt.ylim(1.5, 16)
#
# # plt.legend()          #显示各曲线的图例
# plt.legend(loc=0, numpoints=1)
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
#
# plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
# plt.show()



# sns.set(style="ticks")
# # Create a dataset with many short random walks
# rs = np.random.RandomState(4)
# pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
# pos -= pos[:, 0, np.newaxis]
# step = np.tile(range(5), 20)
# walk = np.repeat(range(20), 5)
# df = pd.DataFrame(np.c_[pos.flat, step, walk],
# columns=["position", "step", "walk"])
# # Initialize a grid of plots with an Axes for each walk
# grid = sns.FacetGrid(df, col="walk", hue="walk", col_wrap=5, size=1.5)
# # Draw a horizontal line to show the starting point
# grid.map(plt.axhline, y=0, ls=":", c=".5")
# # Draw a line plot to show the trajectory of each random walk
# grid.map(plt.plot, "step", "position", marker="o", ms=4)
# # Adjust the tick positions and labels
# grid.set(xticks=np.arange(5), yticks=[-3, 3],
# xlim=(-.5, 4.5), ylim=(-3.5, 3.5))
# # Adjust the arrangement of the plots
# grid.fig.tight_layout(w_pad=1)
# plt.show()





#连接npy
# d1X=np.load('emg/experiment/emgtrainset3_X.npy',allow_pickle=True)
# d1Y=np.load('emg/experiment/emgtrainset3_Y.npy',allow_pickle=True)
# d2X=np.load('emg/experiment/emgtrainset2all_X.npy',allow_pickle=True)
# d2Y=np.load('emg/experiment/emgtrainset2all_Y.npy',allow_pickle=True)
#
# dataX=np.concatenate((d1X,d2X),axis=0)
# print('')
#

#字典f2score
# tmp=np.zeros((4,2))
# f2 = np.zeros((4))
# ta={'fetch': {'precision': 0.8487179487179487, 'recall': 0.9484240687679083, 'f1-score': 0.895805142083897, 'support': 349}, 'pick': {'precision': 0.8628048780487805, 'recall': 0.8062678062678063, 'f1-score': 0.833578792341679, 'support': 351}, 'sw': {'precision': 0.9235127478753541, 'recall': 0.9314285714285714, 'f1-score': 0.9274537695590327, 'support': 350}, 'turn': {'precision': 0.8328267477203647, 'recall': 0.7828571428571428, 'f1-score': 0.8070692194403535, 'support': 350}, 'accuracy': 0.8671428571428571, 'macro avg': {'precision': 0.866965580590612, 'recall': 0.8672443973303572, 'f1-score': 0.8659767308562405, 'support': 1400}, 'weighted avg': {'precision': 0.8669756426829911, 'recall': 0.8671428571428571, 'f1-score': 0.8659322834635674, 'support': 1400}}
#
# tmp[0][0]=ta['fetch']['precision']
# tmp[0][1]=ta['fetch']['recall']
# tmp[1][0]=ta['pick']['precision']
# tmp[1][1]=ta['pick']['recall']
# tmp[2][0]=ta['sw']['precision']
# tmp[2][1]=ta['sw']['recall']
# tmp[3][0]=ta['turn']['precision']
# tmp[3][1]=ta['turn']['recall']
# sum=0
#
# for i in range(tmp.shape[0]):
#     f2[i]=5*tmp[i][0]*tmp[i][1]/(4*tmp[i][0]+tmp[i][1])
#     sum=sum+f2[i]
# print(f2)
# print(sum/4)

# tX=np.load('emg/experiment/fftdata/validset overall_dx.npy',allow_pickle=True)
# tY=np.load('emg/experiment/fftdata/validset overall_dy.npy',allow_pickle=True)
# print('')
# 数组
# name_list =['fetch', 'pick', 'sw', 'turn', 'avg']
# precision = np.array([1, 2, 3, 4, 5])
# recall = np.array([2, 3, 4, 5, 6])
# f1_score = np.array([7, 8, 8, 8, 9])
# acc=9
# x = list(range(len(precision)))
# total_width, n = 0.8, 5
# width = total_width / n
#
# plt.bar(x, precision, width=width, label='precision')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, recall, width=width, label='recall', tick_label=name_list)
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, f1_score, width=width, label='f1-score', tick_label=name_list)
#
# plt.bar( acc, width=width, label='acc', tick_label=name_list)
# plt.legend()
# plt.show()



# ##############################热点图

# label=['fetch', 'pick', 'sw', 'turn']
# print(label[0])
# label.append('test')
# print(label[4])
#
# heat=np.array([335,2,7,5,1,327,5,18,30,0,317,3,16,31,2,301])
# a=np.zeros([4,4])
# for i in range(4):
#     for j in range(4):
#         a[i][j]=heat[i+4*j]/350


# xtick=['a','b','c','d']
# ytick=['a','b','c','d']
# data={}
# for i in range(4):
#     data[xtick[i]] = a[i]
# pd_data=pd.DataFrame(data,index=ytick,columns=xtick)
# print(pd_data)
# fig,ax=plt.subplots(figsize=(17,17))
# sns.heatmap(pd_data,ax=ax,annot=True,fmt='.5g',cmap='OrRd',linewidths=.5,
#             xticklabels=['fetch', 'pick', 'sw', 'turn'],
#             yticklabels=['fetch', 'pick', 'sw', 'turn'],cbar=False,annot_kws={"size":40})
# # plt.xticks(fontsize=50)
# # plt.yticks(fontsize=50)
#
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# font = {'family': 'Times New Roman',
#             'color': 'k',
#             'weight': 'normal',
#             'size': 50,}
# plt.xlabel('Prediction',fontsize=40, color='k') #x轴label的文本和字体大小
# plt.ylabel('Ground truth',fontsize=40, color='k') #y轴label的文本和字体大小
# #设置colorbar的刻度字体大小
# cax = plt.gcf().axes[-1]
# cax.tick_params(labelsize=40)
# #设置colorbar的label文本和字体大小
# cbar = ax.collections[0].colorbar
# # cbar.set_label(r'$NMI$',fontdict=font)
#
# plt.savefig('confusion.eps',dpi=1000)
# plt.show()

# print(sum(a[:][0]))

##################################################ROC曲线
# y = np.array([1, 1, 2, 2])
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# print(metrics.auc(fpr, tpr))





#
# b=[]
# b.append({1,0.2})
# b.append({2,0.7})
# b.append({3,0.5})
# ######################################################点线图
# sns.pointplot(x='dataset size',y='accracy(%)',data=b,palette={"male": "g", "female": "m"},markers=["^", "o"], linestyles=["-", "--"])
# plt.show()
#

########################################################柱状图

# sns.countplot(x="who", data=a, palette="Set2")
# plt.show()
# class fyz:
#     a_name='d'
#     b_name='h'
#     def __init__(self,a,b):
#         self.a_name=a
#         self.b_name=b
#
#
# list_data=[]
#
# list_data.append(fyz('d','d'))
# print(list_data[0].a_name,list_data[0].b_name)
# df = pd.DataFrame({'name': ['tom', 'David', 'mary'], 'age': [18,19,17], 'score': [89,90,59]})
# dataframe=pickle.dumps(df)



# dataframe=pd.DataFrame({'fetch':96,'pick':89.1,'sw':97.2,'turn':75,'avg':92})
# sns.barplot(dataframe)
# plt.show()
# dataframe=pd.DataFrame({'fetch':96,'pick':89.1,'sw':97.2,'turn':75,'avg':92},columns=['1','1','4','2'])
#
# arr=np.random.randn(10, 4).cumsum(0)
# a=np.array([1,4,6.0,8])

# df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
# 					columns=['A', 'B', 'C', 'D'],
# 					index=np.arange(0, 100, 10))
# df.plot()
# plt.show()
#
# r1=3
# r2=6
# r3=90
# r4=87
#
#
# acc=np.array([89.23,45,67,23]).reshape(1,4)
# row=np.array([1,4,56,6])
# acc=np.row_stack((acc,[r1,r2,r3,r4]))
# acc=np.row_stack((acc,[r1,r2,r3,r4]))
# acc=np.row_stack((acc,[r1,r2,r3,r4]))
#
# df = pd.DataFrame(acc,
# 				index=['one', 'two', 'three', 'four'],
# 				# 用pd.index() 创建索引，并赋给 columns 作为列标签
# 				columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus')
#                 )
# # df.plot.bar()
#
#
# #############################################柱状图
# x=['A','B','C','D']
# y=[96,23,65,16]
# df2=pd.DataFrame([96,23,65,16],index=['A','B','C','D'])
# print(df2)
# set1=sns.color_palette('hls',4)
# sns.barplot(x=df2.index.values,y=y,palette=set1)
# plt.show()
