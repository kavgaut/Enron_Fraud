#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)

print "Accuracy of DT (after splitting data into test and training):", acc
count  = 0
count = [count+1 for key in pred if key == 1]
print "How many POIs are predicted for the test set for your POI identifier?", len(count)
print "Total number of persons in test set:", len(pred)
### print "labels for test - just checking", labels_test

### to compare pred and labels and see if there are any true positives at all
print pred
print labels_test
### it turns out pred indices (5, 12, 20 and 22) are predicted as POIs 
### where as the actual labels show indices (23, 25, 26, 28) as POIS. so no true positives in our prediction


print "Printing the precision_score and recall_scores--------:"

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred, labels=range(2))
