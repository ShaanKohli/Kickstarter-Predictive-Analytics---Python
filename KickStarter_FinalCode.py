#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:04:55 2019

@author: SK-MBP
"""


#Please enter the path for the Kick-Starter and Kick-Starter Grading datasets below
#please enter file path where it says "enter traning dataset file path here" or "enter test dataset file path here"
#there are 4 places to enter filepaths
#2 for train (Kickstarter)
#2 for KickStarter-Grading


#------------------------
#--REGRESSION MODEL
#------------------------

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import pandas as pd
import numpy as np
from scipy import stats


#------------------------
#--Data PreProcessing
#------------------------


# Reading Training Dataset
#enter traning dataset file path here
new_df = pd.read_excel(r"/Users/SK-MBP/Documents/MMA/INSY662/IndividualProject/Kickstarter.xlsx")

#Selecting only successful and failed states
new_df = new_df[(new_df.state=="successful") | (new_df.state=="failed")]

#Dropping launch_to_state_change_days due to large # of Na's
new_df = new_df.drop(['launch_to_state_change_days'],axis = 1)

#Then Dropping all Na's from df
new_df = new_df.dropna()

#Function for outliers
def numerical_outliers_correcting(df, limit_z=3):
    # Constrains will contain True or False depending on if it is a value below the threshold.
    constraints = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < limit_z, reduce=False) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constraints], inplace=True)

#Removing Outliers from df
numerical_outliers_correcting(new_df)


#Reading Grading DataSet
#enter test dataset file path here
grad_df = pd.read_excel(r"/Users/SK-MBP/Documents/MMA/INSY662/IndividualProject/Kickstarter-Grading-Sample.xlsx")

#Selecting only successful and failed states
grad_df = new_df[(new_df.state=="successful") | (new_df.state=="failed")]

#Dropping all Na's
grad_df = new_df.dropna()

#Removing Outliers
numerical_outliers_correcting(grad_df)


#Concatenate both training and grading data set for dummy classification
train_test = pd.concat([new_df, grad_df], axis=0)



#Classfiy per week
day_bins = [0,7,14,21,32]

#Classfiy per 8 hour cylcles
hour_bins = [0,8,16,24]

#Creating bins to reduce the number of dummy variables
#deadline_day
train_test['deadline_day'] = pd.cut(train_test['deadline_day'], day_bins)

#deadline_hr
train_test['deadline_hr'] = pd.cut(train_test['deadline_hr'], hour_bins)

#created_at_day
train_test['created_at_day'] = pd.cut(train_test['created_at_day'], day_bins)

#created_at_hr
train_test['created_at_hr'] = pd.cut(train_test['created_at_hr'], hour_bins)

#launched_at_day
train_test['launched_at_day'] = pd.cut(train_test['launched_at_day'], day_bins)

#launched_at_hr
train_test['launched_at_hr'] = pd.cut(train_test['launched_at_hr'], hour_bins)

##Ensuring that amounts are in one currency to draw interpretable comparisons
train_test["goal"] = train_test["goal"]*train_test["static_usd_rate"] 


#Select relevant columns
X=train_test[['goal','country','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]

#Selecting target Variables
y=train_test['usd_pledged']

#Dummifying columns
X = pd.get_dummies(X, columns = ['country','category','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])


#Splitting X between train and test after dummyfying
X_train = X.iloc[:12953][:]
X_test = X.iloc[12953:][:]



#Splitting y between train and testafter dummyfying
y_train = y.iloc[:12953]
y_test = y.iloc[12953:]


#Creating testing dataframes
testing_df = pd.concat([X_test,pd.DataFrame(y_test)], axis = 1)
testing_df = testing_df.dropna()
y_test = testing_df.usd_pledged
X_test = testing_df.iloc[:,:-1]


#Generating optimal feature list
#Not retraining a model just generating the feature list
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state=0)
model = randomforest.fit(X_train,y_train)


from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X_train,y_train)


#Generating list of predictors in descending order of Gini Coefficient
random_forest_output = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
random_forest_output_pred = random_forest_output[random_forest_output['Gini coefficient'] != 0]
random_forest_output_pred1= random_forest_output_pred.sort_values("Gini coefficient", ascending = False)
feature_list1 = np.transpose((random_forest_output_pred1["predictor"]))
feature_list = X[feature_list1][:]



X_reg = X_train[feature_list1][:]
X_reg_test = X_test[feature_list1][:]


#Best Model
model = RandomForestRegressor(random_state = 0, max_features = 23,\
                              max_depth = 8, min_samples_split = 10, \
                              min_samples_leaf = 10, bootstrap = 1,\
                              n_estimators = 100)


model_fit= model.fit(X_reg, y_train)
# Using the model to predict the results based on the test dataset
y_test_pred= model_fit.predict(X_reg_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print("Random Forest Regression Model for usd_pledged has a mse of: ", mse)
#237707339.21886605 for sample grading set



#------------------------
#--CLASSIFICATION MODEL
#------------------------


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import pandas as pd
import numpy as np

#------------------------
#--Data PreProcessing
#------------------------


# Reading Training Dataset
new_df = pd.read_excel(r"/Users/SK-MBP/Documents/MMA/INSY662/IndividualProject/Kickstarter.xlsx")

#Selecting only successful and failed states
new_df = new_df[(new_df.state=="successful") | (new_df.state=="failed")]

#Dropping launch_to_state_change_days due to large # of Na's
new_df = new_df.drop(['launch_to_state_change_days'],axis = 1)

#Then Dropping all Na's from df
new_df = new_df.dropna()

#Function for outliers
def numerical_outliers_correcting(df, limit_z=3):
    # Constrains will contain True or False depending on if it is a value below the threshold.
    constraints = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < limit_z, reduce=False) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constraints], inplace=True)

#Removing Outliers from df
numerical_outliers_correcting(new_df)


#Reading Grading DataSet
grad_df = pd.read_excel(r"/Users/SK-MBP/Documents/MMA/INSY662/IndividualProject/Kickstarter-Grading-Sample.xlsx")

#Selecting only successful and failed states
grad_df = new_df[(new_df.state=="successful") | (new_df.state=="failed")]

#Dropping all Na's
grad_df = new_df.dropna()

#Removing Outliers
numerical_outliers_correcting(grad_df)


#Concatenate both training and grading data set for dummy classification
train_test = pd.concat([new_df, grad_df], axis=0)



#Classfiy per week
day_bins = [0,7,14,21,32]

#Classfiy per 8 hour cylcles
hour_bins = [0,8,16,24]

#Creating bins to reduce the number of dummy variables
#deadline_day
train_test['deadline_day'] = pd.cut(train_test['deadline_day'], day_bins)

#deadline_hr
train_test['deadline_hr'] = pd.cut(train_test['deadline_hr'], hour_bins)

#created_at_day
train_test['created_at_day'] = pd.cut(train_test['created_at_day'], day_bins)

#created_at_hr
train_test['created_at_hr'] = pd.cut(train_test['created_at_hr'], hour_bins)

#launched_at_day
train_test['launched_at_day'] = pd.cut(train_test['launched_at_day'], day_bins)

#launched_at_hr
train_test['launched_at_hr'] = pd.cut(train_test['launched_at_hr'], hour_bins)

##Ensuring that amounts are in one currency to draw interpretable comparisons
train_test["goal"] = train_test["goal"]*train_test["static_usd_rate"] 


#Select relevant columns
X=train_test[['goal','country','category','name_len_clean','blurb_len_clean','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr','create_to_launch_days','launch_to_deadline_days']]

#Selecting target Variables
y=train_test['state']

#Dummifying columns
X = pd.get_dummies(X, columns = ['country','category','deadline_weekday','created_at_weekday','launched_at_weekday','deadline_month','deadline_day','deadline_yr','deadline_hr','created_at_month','created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day','launched_at_yr','launched_at_hr'])


#Splitting X between train and test after dummyfying
X_train = X.iloc[:12953][:]
X_test = X.iloc[12953:][:]



#Splitting y between train and testafter dummyfying
y_train = y.iloc[:12953]
y_test = y.iloc[12953:]


#Creating testing dataframes
testing_df = pd.concat([X_test,pd.DataFrame(y_test)], axis = 1)
testing_df = testing_df.dropna()
y_test = testing_df.state
X_test = testing_df.iloc[:,:-1]



#Generating optimal feature list
#Not retraining a model just generating the feature list
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X_train,y_train)


from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X_train,y_train)


#Generating list of predictors in descending order of Gini Coefficient
random_forest_output = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
random_forest_output_pred = random_forest_output[random_forest_output['Gini coefficient'] != 0]
random_forest_output_pred1= random_forest_output_pred.sort_values("Gini coefficient", ascending = False)
feature_list1 = np.transpose((random_forest_output_pred1["predictor"]))
feature_list = X[feature_list1][:]



X_class = X_train[feature_list1][:]
X_class_test = X_test[feature_list1][:]



# Best Model
model = RandomForestClassifier(random_state = 0, max_features = 123,\
                              max_depth = 118, min_samples_split = 32,\
                              min_samples_leaf = 26, bootstrap = 1,\
                              n_estimators = 100)


#Fitting best model
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import pandas as pd


model_fit = model.fit(X_class, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred= model_fit.predict(X_class_test)
y_test_pred = pd.Series(y_test_pred)

#Accuracy score
from sklearn.metrics import accuracy_score
scores = accuracy_score(y_test, y_test_pred)
print("Random Forest Classification Model for state has an accuracy of: ", np.average(scores))
#Random Forest Classification Model for state has an accuracy of: 0.8145741270253878
#With sample grading data set



