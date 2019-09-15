# -*- coding: utf-8 -*-
"""
Created on Wed May 29 03:05:42 2019

@author: PALLAVI
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values 
y=dataset.iloc[:,2].values

#splitting the dataset into the trainign and testing data
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)'''

'''#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)'''

#fitting the regression model to the dataset
#create your regressor here
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

#predicting a new result with regression model
y_pred=regressor.predict([[6.5]])

#visalizing the Regression Results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('truth or bluff( Regression model)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visalizing the Regression Results (for higher resolution and smoother curve)
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('truth or bluff(Regression Model)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()