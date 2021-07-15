import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio,fs=librosa.load('../audio/sw1-20s5fyz.wav')
audio2,fs2=librosa.load('../audio/turn61-80s5fyz.wav')

D=np.abs(librosa.stft(audio))
S=librosa.feature.melspectrogram(S=D)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', fmax=22050, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

print('')