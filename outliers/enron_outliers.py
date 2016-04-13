#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

## Remove the 'TOTAL' point which is a major outlier for salary and bonus - not a valid data point
data_dict.pop('TOTAL',0)
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

###We need to use data_dict and store salary outliers or bonus outliers into a list

outliers = []
for key in data_dict:
    value = data_dict[key]['salary']
    if value == 'NaN':
        continue
    outliers.append((key, value))
    

outliers = sorted(outliers, reverse = True, key = lambda tup:tup[1])

print "The name of the outlier point wrt salary:", outliers[:2]
    

outliers = []
for key in data_dict:
    value = data_dict[key]['bonus']
    if value == 'NaN':
        continue
    outliers.append((key, value))
    

outliers = sorted(outliers, reverse = True, key = lambda tup:tup[1])

print "The name of the outlier point wrt bonus:", outliers[:2]


print "SKILLING JEFFREY K and LAY KENNETH L"
print "Their BONUS and SALARY"

print data_dict["LAY KENNETH L"]['salary'], data_dict["LAY KENNETH L"]['bonus']
print data_dict["SKILLING JEFFREY K"]['salary'], data_dict["SKILLING JEFFREY K"]['bonus']


    
