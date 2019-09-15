# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:21:50 2019

@author: PALLAVI
"""

'''when we find a error in our output , we back propagate.
we readjust the weights and follow the same procedure again. we do this using GRADIENT DESCENT
in GRADIENT DESCENT we calculate cost after taking y and y hate of all the entries whereas
in STOCHASTIC GRADIENT DESCENT we calculate cost at each step.'''
#we are going to do a classification problem using ANN
#based on given info we are required to predict which customers leave the bank.


#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
datasets=pd.read_csv('Churn_Modelling.csv')
X=datasets.iloc[:,3:13].values
y=datasets.iloc[:,13].values

#handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
#X[:, 2] = labelencoder_X.fit_transform(X[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [2])
#X = onehotencoder.fit_transform(X).toarray()

#splitting into test and training data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#fit the classifier  to the training data set
#create your classifier here

#predicting the test set results
y_pred=classifier.predict(x_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)