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


from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000.0)

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train,labels_train)
print "Training time for SVM:", round(time()-t0,3), "s"

t0 = time()
pred = clf.predict(features_test)
print "Predicting time for SVM:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)

print "Accuracy for SVM:", acc

print 

#print "Prediction for 10th test element:", pred[10]
#print "Prediction for 26th test element:", pred[26]
#print "Prediction for 50th test element:", pred[50]

count = 0

for i in range(0, len(pred)):
	if pred[i] == 1:
		count += 1
print "Total number of predictions as Chris(1):", count

#########################################################

from sklearn.svm import SVC
clf = SVC(kernel='linear', C=10000.0)
#Train
clf.fit(features_train,labels_train)
#predict on Test data
pred = clf.predict(features_test)
#Calculate Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print "Accuracy for SVM:", acc









