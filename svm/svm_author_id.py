#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
#from sklearn import svm
from sklearn.svm import SVC
#clf=SVC(kernel="linear")
clf=SVC(kernel="rbf",C=10000)
from sklearn.metrics import accuracy_score
#clf = svm.SVC(gamma="auto")
print(len(features_train))
print(int(len(features_train)/100))
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
labels_pred=clf.predict(features_test)
t2 = time()
answer=[]
answer.append(labels_pred[10])
answer.append(labels_pred[26])
answer.append(labels_pred[50])
print('answer: ',answer)
print('accuracy: ',accuracy_score(labels_test, labels_pred))
print('training time: ',round(t1-t0,3),'s')
print('prediction time: ',round(t2-t1,3),'s')
print('total class 1: ',sum(labels_pred==1))

