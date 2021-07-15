import matplotlib.pyplot as plt
import  csv
import pandas as pd
import numpy as np
filepath='emg/fetch/fetch (1).csv'

tmp = pd.read_csv(filepath, header=None).values
tmp=tmp[:,1]
# print(tmp)
# plt.plot(tmp)
# plt.show()

x=np.random.uniform(-1, 1, [8, 40]).reshape([8*40])
plt.plot(x)

plt.show()