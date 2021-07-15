#########################################################################################
# 模型 lightgbm
#########################################################################################
from datetime import datetime

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

import lightgbm as lgb

# 准备数据
X = data_set.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, data_set["y"], test_size=0.3,random_state=0)

# 训练
btime = datetime.now()
train_data=lgb.Dataset(X_train,label=y_train)
validation_data=lgb.Dataset(X_test,label=y_test)
params={
    'learning_rate':0.1,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':6,
    'objective':'multiclass',
    'num_class':4,
}
clf=lgb.train(params,train_data,valid_sets=[validation_data])
print ('all tasks done. total time used:%s s.\n\n'%((datetime.now() - btime).total_seconds()))

# 1、AUC
y_pred_pa = clf.predict(X_test)  # !!!注意lgm预测的是分数，类似 sklearn的predict_proba
y_test_oh = label_binarize(y_test, classes= [0,1,2,3])
print ('调用函数auc：', roc_aucx_score(y_test_oh, y_pred_pa, average='micro'))

#  2、混淆矩阵
y_pred = y_pred_pa .argmax(axis=1)
confusion_matrix(y_test, y_pred )

#  3、经典-精确率、召回率、F1分数
precision_score(y_test, y_pred,average='micro')
recall_score(y_test, y_pred,average='micro')
f1_score(y_test, y_pred,average='micro')

# 4、模型报告
print(classification_report(y_test, y_pred))

# 保存模型
# joblib.dump(clf, './model/lgb.pkl')