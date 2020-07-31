# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:21:25 2020

@author: rrust
"""
from sklearn import datasets

iris=datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, stratify=y)
  

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
svc=SVC(kernel = 'rbf',gamma=1.0)
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test)
conf=confusion_matrix(y_test,y_pred)

#for gamma = 10
svc=SVC(kernel = 'rbf',gamma=10.0)
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test)
confg10=confusion_matrix(y_test,y_pred)

#for kernel linear
svc=SVC(kernel = 'linear')
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test)
conflin=confusion_matrix(y_test,y_pred)

svc=SVC(kernel = 'sigmoid')
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test)
confsig=confusion_matrix(y_test,y_pred)

svc=SVC(kernel = 'poly')
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test)
confpoly=confusion_matrix(y_test,y_pred)


