# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:34:18 2019

@author: PALLAVI
"""

'''
in our sample problem we are required to find how profit varies with respect
 to all the independednt variable so that the company can invest in such a
 manner that  ,say, the profit is maximum with minimum sales or somthing
 like that
 '''
#in MULTIPLE REGRESSION:
#y=b0 + b1*x1 + b2*x2+........+bn*xn
 #y=dependent variable i.e. profit
 #xi independent variable
 #x1 for r&d nd b1
 #x2 for admin and b2
 #x3 for marketing and b3
#state is categorical variable so we create dummy variables for them
 #as we have two states we will create two dummy columns but we include only one ..
#so we can have b4 and D1 where d1 is summy for newyork ..
#when it is 1 it is newyork's equation otherwise that of california

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#handling categorical data:states
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
x[:,3]=le.fit_transform(x[:,3])
one=OneHotEncoder(categorical_features=[3])
x=one.fit_transform(x).toarray()

#avoiding dumy variable trap
x=x[:,1:]

#traing and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

# predicting the dataset
y_pred=regressor.predict(x_test)

'''now we have our model but it contains all the variables .there may be some variables which do not play 
any role at all but are there without any use so we will try to remove them using backward Elimination'''

import statsmodels.formula.api as sm
x=np.append(np.ones((50,1)).astype(int),values=x,axis=1)
#setp:2  fit the full model with the predictors
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
#step:3   checking the p value i.e. considering the predictor with highest p value
regressor_OLS.summary()
#step:4  removing the predictor with highest p value
x_opt=x[:,[0,1,3,4,5]]
#step:5 fitting the model without this predictor
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt=x[:,[0,3,4,5]]
#step:5 fitting the model without this predictor
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt=x[:,[0,3,5]]
#step:5 fitting the model without this predictor
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt=x[:,[0,3]]
#step:5 fitting the model without this predictor
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

