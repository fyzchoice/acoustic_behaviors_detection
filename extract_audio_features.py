import librosa
import matplotlib.pyplot as plt
import librosa.display
import wave
import numpy as np
# audio,sr=librosa.load('E://Desktop/acoustic signal program/audio_lstm/audio/yyy3n3h.wav')



# X = librosa.stft(audio,2048,2000,2048,'hamm')
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure()
# librosa.display.specshow(Xdb, sr=sr*2, x_axis='time', y_axis='hz')
# plt.colorbar()
# plt.show()

#mfcc
# mfccs=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13)
# print(mfccs.shape)

# while 未到文件尾：
#     node1=locat1
#     if(开始读数标记)：


# ================================
#短时能量
# wlen=2048
# inc=1960
# f=wave.open('audio/turn61-80s5fyz.wav','rb')
# params=f.getparams()
# nchannels,sampwidth,framerate,nframes=params[:4]
# str_data=f.readframes(nframes)
# print(nchannels,sampwidth,framerate,nframes)
#
# wave_data = np.fromstring(str_data, dtype=np.short)
# wave_data = wave_data*1.0/(max(abs(wave_data)))
# # print(wave_data[:10])
# time = np.arange(0, wlen) * (1.0 / framerate)
# signal_length=len(wave_data) #信号总长度
# if signal_length<=wlen: #若信号长度小于一个帧的长度，则帧数定义为1
#         nf=1
# else: #否则，计算帧的总长度
#         nf=int(np.ceil((1.0*signal_length-wlen+inc)/inc))
# pad_length=int((nf-1)*inc+wlen) #所有帧加起来总的铺平后的长度
# zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
# pad_signal=np.concatenate((wave_data,zeros)) #填补后的信号记为pad_signal
# indices=np.tile(np.arange(0,wlen),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(wlen,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
# print(indices[:2])
# indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
# frames=pad_signal[indices] #得到帧信号
# a=frames[30:31]
# print(a[0])
# windown=np.hanning(2048)
# b=a[0]*windown
# c=np.square(b)
# plt.figure(figsize=(10,4))
# plt.plot(time,c,c="g")
# plt.grid()
# plt.show()

#分帧函数
wlen=44100//4
inc=3*44100//16
f = wave.open(r"audio/sw1-20s5fyz.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
str_data = f.readframes(nframes)
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data = wave_data*1.0/(max(abs(wave_data)))
signal_length=len(wave_data) #信号总长度

if signal_length<=wlen: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
else:                 #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-wlen+inc)/inc))
pad_length=int((nf-1)*inc+wlen) #所有帧加起来总的铺平后的长度
zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
pad_signal=np.concatenate((wave_data,zeros)) #填补后的信号记为pad_signal

indices=np.tile(np.arange(0,wlen),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(wlen,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*wlen长度的矩阵

indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
frames=pad_signal[indices] #得到帧信号
a=frames[30:31]
print(a[0])

