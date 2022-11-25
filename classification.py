
#_____________________Titanic survival classification with svm method___________________________


#load the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions

#Load the Data
test = pd.read_csv('/home/veunex/a/MachineLearning/classification/titanic/test.csv')
train = pd.read_csv('/home/veunex/a/MachineLearning/classification/titanic/train.csv')
result = pd.read_csv('/home/veunex/a/MachineLearning/classification/titanic/gender_submission.csv')

#fix NaN data
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("C")
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Embarked"] = test["Embarked"].fillna("C")
test["Fare"] = test["Fare"].fillna(test['Fare'].median())

#label non-numeric columns
train.Sex=train.Sex.replace(['male','female'],[0,1])
train.Embarked=train.Embarked.replace(['S','C','Q'],[0,1,2])
test.Sex=test.Sex.replace(['male','female'],[0,1])
test.Embarked=test.Embarked.replace(['S','C','Q'],[0,1,2])


#define train x and y
Y_train = train['Survived']
X_train = train.drop(['Name','Survived','PassengerId','Ticket','Cabin'], axis = 1)
X_test = test.drop(['Name','PassengerId','Ticket','Cabin'], axis = 1)



Y_result = result['Survived']

X_train = X_train.astype(float)
Y_train = Y_train.astype(float)
X_test = X_test.astype(float)


corr_relation=train.corr()
sns.heatmap(corr_relation,annot=True,cmap="Blues")
plt.show()


#fit the model
clf1 = SVC(kernel='rbf').fit(X_train,Y_train)
clf2 = SVC(kernel='linear').fit(X_train,Y_train)
clf3 = SVC(kernel ='poly').fit(X_train , Y_train)
clf4 = SVC(kernel ='sigmoid').fit(X_train , Y_train)

#predict the survival
pred1 = clf1.predict(X_test)
pred2 = clf2.predict(X_test)
pred3 = clf3.predict(X_test)
pred4 = clf4.predict(X_test)


#check the accuracy
print('rbf score' ,accuracy_score(Y_result,pred1))
print('linear score',accuracy_score(Y_result,pred2))
print('poly score',accuracy_score(Y_result,pred3))
print('sigmoid score',accuracy_score(Y_result,pred4))

#save the result
submission = pd.DataFrame({"PassengerId":test["PassengerId"],
                           'survived linear': pred3,
                           'actual result' :Y_result
                           })
submission.to_csv("titanic_result.csv", index=False)


