# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 07:03:12 2022

@author: Harshitha
"""

import os
os.getcwd()
import pandas as pd
os.chdir("C:/Users/Harshitha/OneDrive/Documents/python code")
mydata=pd.read_csv("titanic.csv")
titanic=mydata.drop(["PassengerId","Name","SibSp","Parch","Ticket","Fare","Cabin","Pclass"],axis=1)
titanic.isnull().sum()
titanic.Age.median()
titanic["Age"].fillna(titanic["Age"].median(),inplace=True)
titanic["Embarked"].fillna("S",inplace=True)
y=titanic[["Survived"]]
x=titanic.drop(["Survived"],axis=1)
titanic["Embarked"].value_counts()
x= pd.get_dummies(x)
print(x)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20,random_state = 0)
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest) 
y_pred
cm = confusion_matrix(ytest, y_pred)
cm
from sklearn.metrics import accuracy_score
a1 = accuracy_score(ytest,y_pred)
print("Accuracy score : {:.2f}%".format(a1*100))



