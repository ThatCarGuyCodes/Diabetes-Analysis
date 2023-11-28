#RESULTS AT BOTTOM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("diabetes.csv")

print((df['diabetes']==0).sum())
print((df['diabetes']==1).sum())

df0 = df[df['diabetes']==0]
df1 = df[df['diabetes']==1]

print(df0['glucose'].mean())
print(df1['glucose'].mean())

print(df0['bloodpressure'].mean())
print(df1['bloodpressure'].mean())

df1['glucose'].hist()
#plt.show()

y=df['diabetes']
x=df.drop('diabetes', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# GNB
bmodel = GaussianNB()
bmodel.fit(x_train, y_train)

print("Gaussian Bayesian", bmodel.score(x_test, y_test))

y_pred = bmodel.predict(x_test)
print(y_pred)
print(y_test)
cm1 =confusion_matrix(y_test, y_pred)
sns.heatmap(cm1,annot=True)
plt.show()
# DT
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Decision Tree",dt.score(x_test, y_test))

predicted = dt.predict(x_test)
cm1 =confusion_matrix(y_test, predicted)
sns.heatmap(cm1,annot=True)
plt.show()
# KNN
n=31
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_train, y_train)
print("KNN",n, knn.score(x_test,y_test))

pred=knn.predict(x_test)
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True)
plt.show()
'''
num = len(y_pred)
print(num)
numAcceptable = 0
for i in range(len(y_pred)):
    if (abs(y_pred[i] - y_test[i]) <=3):
        numAcceptable +=1

print("Accuracy: ", numAcceptable/num)

from math import sqrt
mse = mean_squared_error(y_true, y_pred)
print(sqrt(mse))
'''
# LR
rm = LogisticRegression()
rm.fit(x_train,y_train)

accuracy2 = rm.score(x_test,y_test)
print("logistic regression",accuracy2*100)

predicted2 = rm.predict(x_test)
cm=confusion_matrix(y_test, predicted2)
sns.heatmap(cm,annot=True)
plt.show()

#GNB: 93.5
#GNB: 90
#GNB: 93
#GNB: 91
#GNB: 93.5
#DT: 91
#DT: 90.5
#DT: 92.5
#DT: 93.5
#KNN 31: 91.5
#KNN 31: 90.5
#KNN 31: 94
#LR: 93
