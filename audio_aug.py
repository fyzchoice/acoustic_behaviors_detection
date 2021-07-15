import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy import signal
import numpy as np

Fs=44100            #音频采样频率
x,fs=librosa.load('audio/sw1-20s5fyz.wav',sr=Fs,mono=False)
t=len(x)
print('音频长度',t/(fs))


plt.specgram(s)
plt.show()