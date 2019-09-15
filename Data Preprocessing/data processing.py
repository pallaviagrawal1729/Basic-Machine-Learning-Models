# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#these are three most important libraries that we need to import
import numpy as np
import matplotlib.pyplot as py
import pandas as pd

'''after importing these libraries we just have to explore the file in which our data set resides

after this now we have to create a dataframe of datasets from spreadsheet file in the python
for this we use the pandas's read_csv'''
datasets=pd.read_csv('Data.csv')

'''after creating datasets the variable can be read from variable explorer
now we need to differently create variables for independent and dependent vectors
in this the first semi colon indicates the row i.e. starting row:ending row+1
in this second semi colon indicates columns i.e. starting column:ending column+1
and lastly the .values indicates that we have to include all the values.
x=independent variable'''
x=datasets.iloc[:,:-1].values

#y=dependent variable
y=datasets.iloc[:,3].values

'''in real world a lot of data is missing from different columns i.e. a part of single entity.
we may think of deleteing the whole row but we cannot do this because it may contain some crucial information 
so what we do is take mean of all the columns of which data is missing and place it in that column.
this is the way we deal with MISSING DATA'''
from sklearn.impute import SimpleImputer as imp
#impute is Transformers for missing value imputation
#simpleImputer is Imputation transformer for completing missing values.
imputer=imp(missing_values=np.nan,strategy='mean',copy=True) 
#use ctrl+I for reading about rest
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

'''now one more important thing is that in machine learning we will deal with mathematical data so we need to
do something for the data that is in character format. in our data set we see two columns to have character data
Country and Purchased, these are Categorical variables i.e. they can be cateorized for say purchased can be 1.Yes 2.No
Country can be 1.Germany 2.Spain 3.France
so our now new motive is to encode Catgorical variables in mathematical or numerical format'''
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
x[:,0]=label.fit_transform(x[:,0])
#here label.fit_transform(x[:,0]) returns array after transforming 0th column and replaces it with x[:,0]
y=label.fit_transform(y)
'''now the problem is that it will assign 0 1 and 2 to different countries but this will create prefrential order 
this order would be fine if in place of country we had size so it could be small medium and large
but between countires spain > germany > france is invalid
so to resolve this we use another library'''
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
'''from sklearn.preprocessing import OrdinalEncoder
ordinal=OrdinalEncoder()
x=ordinal.fit_transform(x)'''
#OneHotEncoder Encode categorical integer features as a one-hot numeric array.
#categorical_features determine which array indice is the feature that we want to categorize
x=onehotencoder.fit_transform(x).toarray()
#onehotencoder.fit_transform(x) transfroms x i.e. 0 th column and then we convert it to array and place it in x
#also note that onehotencoder.fit_transform(x).toarray().toarray() contains whole x but transforms only 0th column
'''
these variables are called DUMMY VARIABLESt
NOW THE X IS:
fr. ge. sp. age salary
1	0	0	44	72000
0	0	1	27	48000
0	1	0	30	54000
0	0	1	38	61000
0	1	0	40	63777.8
1	0	0	35	58000
0	0	1	38.7778	52000
1	0	0	48	79000
0	1	0	50	83000
1	0	0	37	67000

'''
#A VERY IMPORTANT NOTE:
#we do not need to use onehotencoder for y because it is dependent variable so python will automaically understand
#that it is categorical variable 
'''
now we need to have different training and test datasets, we require them so that we can test our model if it
predicts the right result or not on what it has learnt using training datasets '''
#generally we take value of ransokm_state to be 0 or 42
#random_state makes sure that the same list is generated whenever we execute our script
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,shuffle=True,test_size=4)
'''now we have two columns with neumerical data. both have different scales that varies enormously when these
 two scales are comapred. for say the age varies from 27 to 50 whereas salary varis from 52000 to 83000
 in many ML algorithms we need to calculate eucledian distances i.e
 sqrt( (x2-x1)**2 + (y2-y1)**2 )and 
 if we take two rows and calculate its (column difference) **2 we will find that one will dominate on other as if 
 other does not exist so to resolve this issue we need to have FATURE SCALING
 
 STANDARDIZATION:
     Xstand= X - mean(X)/standard deviation(X)
             
 NORMALIZATION:
     Xnorm= X- min(X)/max(X)-min(X)
 
 '''
#also note that here we have done scaling only for x because it varies hugely but not for y because of vice 
#versa case
from sklearn.preprocessing import StandardScaler
#creating object of class
sc=StandardScaler()
#we fit and trasform training variables 
x_train=sc.fit_transform(x_train)
#and on the basis which we fit the training variables we transform the test vairables.
x_test=sc.transform(x_test)



'''now when preparing Data Preprocessing Template we will not consider the following:
    missing data
    categorical data
    feature scaling
we will apply feature scaling only in comments because some of the data requires it while others do not for say
our model did not require it for y'''


 



















