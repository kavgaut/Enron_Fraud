#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

#THE FOLLOWING CHUNK OF CODE IS TO GE THE DT up and TRAINING AND TO PRINT TEST ACCURACY
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)


pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc_train = accuracy_score(labels_train, pred_train)
acc_test = accuracy_score(labels_test, pred_test)

print "Accuracy of DT training:", acc_train
print "Accuracy of DT test:", acc_test

#THIS CODE WILL USE THE FEATURE_IMPORTANCES ATTRIBUTE TO GET A LIST OF RELATIVE IMPORTANCES OF ALL FEATURES AND PRINTS IF IMP>0.2

problem = 33614
filtered = clf.feature_importances_
print len(filtered)
for i in range(len(filtered)):
    if (filtered[i] > 0.2):
        problem = i
        print i, filtered[i]
        
        
#FIGURE OUT THE WORD THATS CAUSING THE PROBLEM
print vectorizer.get_feature_names()[problem]