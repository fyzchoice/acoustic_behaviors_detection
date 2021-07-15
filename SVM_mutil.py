import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from scipy.interpolate import lagrange, interp1d
import init_args
import process_emg
import tensorflow as tf
from datetime import datetime
init=init_args.init_args()

args=init.getargs()

features=args[0]
time_step=int(args[1]/features)
print('arg:',args[0],args[1],time_step)
dataX,dataY=process_emg.get4dataall(process_emg.filespath[0],process_emg.filespath[1],process_emg.filespath[2],process_emg.filespath[3])
dataY=np.array(dataY)
dataX=np.array(dataX)
dataX=tf.keras.preprocessing.sequence.pad_sequences(dataX,maxlen=args[1],padding='post')
dataX=np.reshape(dataX,(dataX.shape[0],-1))


trainX,testX,trainY,testY=train_test_split(dataX,dataY, test_size=0.2, random_state=0)
# 二进制化输出

n_classes = dataY.shape[1]
random_state = np.random.RandomState(0)

btime=datetime.now()
print('begin:')
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
y_score = classifier.fit(trainX, trainY).decision_function(testX)
etime=datetime.now()

label = ['fetch', 'pick', 'sw', 'turn']
y_ture = testY
ture = []
pre = []
for i in range(len(y_score)):
    lture = label[np.argmax(y_ture[i])]
    lpre = label[np.argmax(y_score[i])]
    ture.append(lture)
    pre.append(lpre)


print('用时：',etime-btime)
t = classification_report(ture, pre, target_names=['fetch', 'pick', 'sw', 'turn'])
print(t)
ta=classification_report(ture, pre, target_names=['fetch', 'pick', 'sw', 'turn'],output_dict=True)
print(ta)
tmp=np.zeros((4,2))
f2 = np.zeros((4))
tmp[0][0]=ta['fetch']['precision']
tmp[0][1]=ta['fetch']['recall']
tmp[1][0]=ta['pick']['precision']
tmp[1][1]=ta['pick']['recall']
tmp[2][0]=ta['sw']['precision']
tmp[2][1]=ta['sw']['recall']
tmp[3][0]=ta['turn']['precision']
tmp[3][1]=ta['turn']['recall']
sum=0

for i in range(tmp.shape[0]):
    f2[i]=5*tmp[i][0]*tmp[i][1]/(4*tmp[i][0]+tmp[i][1])
    sum=sum+f2[i]
print(f2)
print(sum/4)
print('above f2:',f2,sum/4)




# 为每个类别计算ROC曲线和AUC
fpr = dict()        ### 假正例率
tpr = dict()        ### 真正例率
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

########################### 计算宏平均ROC曲线和AUC ###########################
### 每个二分类，各自算各自的，再综合
fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

########################### 计算微平均ROC曲线和AUC ###########################
### 先综合每个二分类的，再综合
# 汇总所有FPR
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
print(all_fpr.shape)        # (42,)

# 然后再用这些点对ROC曲线进行插值
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    #### 把每个 二分类 结果 加起来了
    # mean_tpr += interp(all_fpr, fpr[i], tpr[i])     ### 版本不同
    f = interp1d(fpr[i], tpr[i])                  ### 这两句和上面一句是一个作用
    mean_tpr += f(all_fpr)

# 最后求平均并计算AUC
mean_tpr /= n_classes
print(mean_tpr)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

########################### 绘制所有ROC曲线 ###########################
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','#990036'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()