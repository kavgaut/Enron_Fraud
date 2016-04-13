#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "No of people in the dataset: ", len(enron_data)

print "No of features for each person: ", len(enron_data["SKILLING JEFFREY K"])

count = 0
count = [count+1 for key in enron_data if enron_data[key]["poi"] == 1]
print "Number of POIs identified: ", len(count)

num_lines = sum(1 for line in open("../final_project/poi_names.txt"))
pois = num_lines - 2
print "No of POIs from the ../final_project/poi_names.txt file:", pois

print "Total value of stock belonging to James Prentice is: ", enron_data["PRENTICE JAMES"]['total_stock_value']

print "No of email messages from Wesley Colwell to POIs:", enron_data["COLWELL WESLEY"]['from_this_person_to_poi']

print "Value of stock options exercised by Jeffrey Skilling:", enron_data["SKILLING JEFFREY K"]['exercised_stock_options']

##Last set of questions after netflix documentary
print "who took money home the most?"
print enron_data["SKILLING JEFFREY K"]['total_payments']
print enron_data["LAY KENNETH L"]['total_payments']
print enron_data["FASTOW ANDREW S"]['total_payments']

#How many folks in this dataset have a quantified salary? What about a known email address?
count = 0
count = [count+1 for key in enron_data if enron_data[key]["salary"] != 'NaN']
print "Number of folks with a quantified salary: ", len(count)

count = 0
count = [count+1 for key in enron_data if enron_data[key]["email_address"] != 'NaN']
print "Number of folks with a known email address: ", len(count)

#total payments being NaN and percentage of those wrt total dataset
count = 0
count = [count+1 for key in enron_data if enron_data[key]["total_payments"] == 'NaN']
print "Number of folks with unknown total payments: ", len(count)

#Number of POIs for which total payments is missing
count = 0
count = [count+1 for key in enron_data if ((enron_data[key]["poi"] == 1) and (enron_data[key]["total_payments"] == 'NaN'))]
print "Number of POIs with unknown total payments: ", len(count)