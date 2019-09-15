# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:46:27 2019

@author: PALLAVI
"""

'''
Regression models (both linear and non-linear) are used for predicting a real value, like salary for
 example. If your independent variable is time, then you are forecasting future values, otherwise your model 
 is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random 
 Forests Regression.

In this part, you will understand and learn how to implement the following Machine Learning Regression models:

Simple Linear Regression
Multiple Linear Regression
Polynomial Regression
Support Vector for Regression (SVR)
Decision Tree Classification
Random Forest Classification
'''

#SIMPLE LINEAR REGRESSION:
'''
y=b0 + x1*b1

y=independent variable(IV)
x1=dependent variable(DV)
b1= per unit change in y w.r.t x1
b0=constant'''

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
datasets=pd.read_csv('Salary_Data.csv')
x=datasets.iloc[:,:-1].values            #independent=year
y=datasets.iloc[:,1].values            #dependent=salary

#splitting into test and training data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#in linear regression we do not need to use feature scaling ,library that we will use will take care of that
'''#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)'''


#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
##regressor.score
#regressor is our trained model on our training set'

#predicting the test set result
#we will create a vector to store all the predicted salaries
y_pred=regressor.predict(x_test)

#visualizing the training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train))
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.title('salary vs experience (training set)')
plt.show()

#visulaizing the test set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred)
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.title('salary vs experience (test set)')
plt.show()






















