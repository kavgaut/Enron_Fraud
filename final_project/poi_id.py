#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from numpy import mean
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree
from sklearn.grid_search import GridSearchCV

def test_clf(grid_search, features, labels, parameters, iterations=100):
	precision, recall = [], []
	for iteration in range(iterations):
		features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=iteration)
		grid_search.fit(features_train, labels_train)
		predictions = grid_search.predict(features_test)
		precision = precision + [precision_score(labels_test, predictions)]
		recall = recall + [recall_score(labels_test, predictions)]
		if iteration % 10 == 0:
			sys.stdout.write('.')
	print '\nPrecision:', mean(precision)
	print 'Recall:', mean(recall)
	best_params = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print '%s=%r, ' % (param_name, best_params[param_name])

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.


### The first feature must be "poi".
features_list = ['poi'] # This is the target label.

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

num_data_points = len(data_dict)
num_data_features = len(data_dict[data_dict.keys()[0]])

num_poi = 0
for dic in data_dict.values():
	if dic['poi'] == 1: num_poi += 1

print "Data points: ", num_data_points
print "Features: ", num_data_features
print "POIs: ", num_poi

employee_names = []
for employee in data_dict:
    employee_names.append(employee)
employee_set = set(employee_names)

#Sort alphabetically and visually inspect
employee_names.sort()
#print(employee_names) 

###Remove the outlier "TOTAL" from the original dataset
data_dict.pop('TOTAL', 0)
print "Data points after removing outlier", len(data_dict) #144 records as expected


### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.

my_dataset = data_dict

## ADDITIONAL FEATURE 1
## Add the fraction of emails from poi and to poi as an email feature
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0
    
    if (poi_messages == 'NaN' or all_messages == 'NaN'):
        return 0
    else:
        fraction = poi_messages*1.0/all_messages
        
    return fraction
    
    
###Actual creation of feature happens here    
for name in my_dataset:

    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi
    

## ADDITIONAL FEATURE 2 
"""
And as a financial feature, we can add the total assets value which intuitively 
will be the sum of 'salary', 'bonus', 'total_stock_value' """

for name in my_dataset:
	data_point = my_dataset[name]
	if (all([	data_point['salary'] != 'NaN',
			    data_point['exercised_stock_options'] != 'NaN',
				data_point['total_stock_value'] != 'NaN',
				data_point['bonus'] != 'NaN'
			])):
		data_point['wealth'] = sum([data_point[each] for each in ['salary',
														   'exercised_stock_options',
														   'total_stock_value',
														   'bonus']])
	else:
	    data_point['wealth'] = 'NaN'

### Now add these above features + some more additional features to the feature_list
features1 = features_list + ['fraction_from_poi',
							   'fraction_to_poi',
							   'shared_receipt_with_poi',
							   'expenses',
							   'loan_advances',
							   'long_term_incentive',
							   'restricted_stock',
							   'salary',
							   'total_stock_value',
							   'exercised_stock_options',
							   'total_payments',
							   'bonus',
							   'wealth']

print ""
print "Two new features succesfully added to the feature list - 'fraction_from_poi', 'fraction_to_poi' and 'wealth'"
print ""
print "Selected Feature list - before Feature_Selection", features1

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features1, sort_keys = True)
labels, features = targetFeatureSplit(data)


### We do not know yet if feature scaling and feature filering using kbest will benefit our model yet.
### But lets try it anyway

# Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# K-best features - choosing 6 features for a trial
k_best = SelectKBest(k=6)
k_best.fit(features, labels)

result_list = zip(k_best.get_support(), features1[1:], k_best.scores_)
result_list = sorted(result_list, key=lambda x: x[2], reverse=True)
#print "K-best features - i.e. top 6 features selected:", result_list

"""
OUTPUT:
K-best features - i.e. top 6 features selected: 
[(True, 'exercised_stock_options', 25.097541528735491), 
(True, 'total_stock_value', 24.467654047526391), 
(True, 'bonus', 21.060001707536578), 
(True, 'wealth', 19.457343207083316), 
(True, 'salary', 18.575703268041778), 
(True, 'fraction_to_poi', 16.641707070468989), 
(False, 'long_term_incentive', 10.072454529369448), 
(False, 'restricted_stock', 9.3467007910514379), 
(False, 'total_payments', 8.8667215371077805), 
(False, 'shared_receipt_with_poi', 8.7464855321290802), 
(False, 'loan_advances', 7.2427303965360172), 
(False, 'expenses', 6.234201140506757), 
(False, 'fraction_from_poi', 3.2107619169667667)]
"""

## 6 best features chosen by SelectKBest
features2 = features_list + ['exercised_stock_options',
							   'total_stock_value',
							   'bonus',
							   'wealth',
							   'salary',
							   'fraction_to_poi']							   
print ""
print "Features succesfully scaled and reduced to Top 6"
print ""
print "Selected Feature list - By KBest Feature selection", features2

## Finally my features that sound intuitive to me
my_features = features_list + ['wealth',
							   'fraction_to_poi',
							   'fraction_from_poi',
							   'shared_receipt_with_poi']							   
print ""
print "Selected Feature list - By Intuition", my_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

"""
Classifier Number 1 - Naive Bayes
"""
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# parameters = {}
# grid_search = GridSearchCV(clf, parameters)
# print '\nGaussianNB:'
# test_clf(grid_search, features, labels, parameters)

"""
OUTPUT:
GaussianNB:
..........
Precision: 0.390936507937
Recall: 0.25930952381
"""
#param grid from http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html

"""
***This is the final Chosen Classifier becaue of a sizeable and comparable Precision and Recall values***
Classifier Number 2 - Decision Tree
"""
from sklearn import tree
clf = tree.DecisionTreeClassifier()

parameters = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 10, 20],
              'max_depth': [None, 2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              'max_leaf_nodes': [None, 5, 10, 20]}
grid_search = GridSearchCV(clf, parameters)
print '\nDecisionTree:'
test_clf(grid_search, features, labels, parameters)

"""
FINAL OUTPUT with my_features without feature_scaling:
DecisionTree:
..........
Precision: 0.392575396825
Recall: 0.416892857143
criterion='entropy', 
max_depth=10, 
max_leaf_nodes=5, 
min_samples_leaf=1, 
min_samples_split=2, 
"""

"""
Classifier Number 3 - AdaBoost Classifier - Ensembles of trees
"""
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()
# parameters = {'n_estimators': [10, 20, 30, 40, 50],
#               'algorithm': ['SAMME', 'SAMME.R'],
#               'learning_rate': [.5,.8, 1, 1.2, 1.5]}
# grid_search = GridSearchCV(clf, parameters)
# print '\nAdaBoost:'
# test_clf(grid_search, features, labels, parameters)

"""
OUTPUT:
AdaBoost:
..........
Precision: 0.401472222222
Recall: 0.228115079365
algorithm='SAMME', 
learning_rate=0.5, 
n_estimators=30, 
"""

"""
Classifier Number 4 - K nearest Neighbor Classifier
"""
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()
# parameters = {'algorithm' : ['auto', 'kd_tree'],
# 			  'leaf_size' : [30, 40], 
# 			  'metric' : ['minkowski'],
#               'n_neighbors' : [2, 3, 4, 5, 6, 7], 
#               'p' : [2],
#               'weights' : ['uniform','distance']}
# grid_search = GridSearchCV(clf, parameters)
# print '\nKNeighborsClassifier:'
# test_clf(grid_search, features, labels, parameters)

"""
OUTPUT:
KNeighborsClassifier:
..........
Precision: 0.3215
Recall: 0.158333333333
algorithm='auto', 
leaf_size=30, 
metric='minkowski', 
n_neighbors=2, 
p=2, 
weights='uniform', 
"""

"""
Classifier Number 5 - Principal Component Analysis and Support Vector Machine
"""
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('svm', SVC())]
clf = Pipeline(estimators)
parameters = dict(reduce_dim__n_components=[2, 5], svm__C=[0.1, 10, 100])
grid_search = GridSearchCV(clf, parameters)
print '\nPCA and SupportVectorMachine:'
test_clf(grid_search, features, labels, parameters)

"""
OUTPUT:
FIRST PCA, THEN SupportVectorMachine:
..........
Precision: 0.0
Recall: 0.0
reduce_dim__n_components=2, 
svm__C=0.1,
"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!	

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
print "************************"
test_classifier(clf, my_dataset, my_features)

#*************************************************************************************************************************
"""
Classifier Number 1 - Naive Bayes
"""
# OUTPUT:
"""
GaussianNB()
	Accuracy: 0.84364	Precision: 0.42879	Recall: 0.28450	F1: 0.34205	F2: 0.30503
	Total predictions: 14000	True positives:  569	False positives:  758	False negatives: 1431	
	True negatives: 11242
"""

#*************************************************************************************************************************
#***This is the final Chosen Classifier because of a sizeable and comparable Precision and Recall values
"""
Classifier Number 2 - Decision Tree - WINNER
"""
# OUTPUT:
"""
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
	Accuracy: 0.88180	Precision: 0.40733	Recall: 0.40000	F1: 0.40363	F2: 0.40145
	Total predictions: 10000	True positives:  400	False positives:  582	False negatives:  600	
	True negatives: 8418
"""

#*************************************************************************************************************************
"""
Classifier Number 3 - AdaBoost Classifier - Ensembles of trees
"""
# OUTPUT:
"""
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None)
	Accuracy: 0.82550	Precision: 0.34521	Recall: 0.24700	F1: 0.28796	F2: 0.26190
	Total predictions: 14000	True positives:  494	False positives:  937	False negatives: 1506	
	True negatives: 11063
"""

#*************************************************************************************************************************
"""
Classifier Number 4 - K nearest Neighbor Classifier
"""
# OUTPUT:
"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
	Accuracy: 0.85264	Precision: 0.38628	Recall: 0.05350	F1: 0.09398	F2: 0.06464
	Total predictions: 14000	True positives:  107	False positives:  170	False negatives: 1893	
	True negatives: 11830
"""

#*************************************************************************************************************************
"""
Classifier Number 5 - PCA & Support Vector machines
"""
# OUTPUT:
"""
Got a divide by zero when trying out: Pipeline(steps=[('reduce_dim', PCA(copy=True, n_components=None, whiten=False)), ('svm', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
Precision or recall may be undefined due to a lack of true positive predicitons.
"""


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features)