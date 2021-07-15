import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 选取一些特征作为我们划分的依据
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 填充缺失值
x['age'].fillna(x['age'].mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

dt = DictVectorizer(sparse=False)

print(x_train.to_dict(orient="record"))

# 按行，样本名字为键，列名也为键，[{"1":1,"2":2,"3":3}]
x_train = dt.fit_transform(x_train.to_dict(orient="record"))

x_test = dt.fit_transform(x_test.to_dict(orient="record"))

# 使用决策树
dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

dt_predict = dtc.predict(x_test)

print(dtc.score(x_test, y_test))

print(classification_report(y_test, dt_predict, target_names=["died", "survived"]))

# 使用随机森林

rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

rfc_y_predict = rfc.predict(x_test)

print(rfc.score(x_test, y_test))

print(classification_report(y_test, rfc_y_predict, target_names=["died", "survived"]))