# -*- coding: utf-8 -*-
"""
Created on Sat May 18 23:25:34 2019

@author: PALLAVI
"""

'''
according to tutorial we predict the values after split based on the mean value of all the data
that lies with in that split region

'''
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

#fitting the Decision Tree regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting a new result with Decision Tree regression model
y_pred=regressor.predict([[6.5]])

#visalizing the Decision Tree Regression Results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('truth or bluff(Decision Tree Regression model)')
plt.xlabel('level1')
plt.ylabel('salary')


#visalizing the Regression Results (for higher resolution and smoother curve)
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('truth or bluff(Decisionegression Model)')
plt.xlabel('level')
plt.ylabel('salary')

