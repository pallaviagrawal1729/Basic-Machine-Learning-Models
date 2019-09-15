# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:18:50 2019

@author: PALLAVI
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
'''suppose it take around four year fr turning a region manager into  a partner
and suppose a guy comes for intervies and says that he has been region manager of this company and had a 
salary of this much and now wants a salary of some amount x.
you want to check whether he is saying truth about his salary of bluffing it.
so he is in mid of level 6 and 7 i.e. 6.5 so we are basically required the salary at level 6.5'''

x=dataset.iloc[:,1:2].values #1:2 is done bcoz x must be a matrix.
y=dataset.iloc[:,2].values

'''we are not going to split our ataset into training and testing data bcoz our data is so small'''

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting polynomial regression to the model
from sklearn.preprocessing import PolynomialFeatures
#poly_reg will be a variablethat will be used for a polynomial type containing our 
#independent variable and its powers
poly_reg=PolynomialFeatures(degree=4) #we check degress as per our model by hit andtrial
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#visualizing the Linear Regression model
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('truth or bluff(Linear Regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualizing the Polynomial Regression model
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('truth or bluff(Polynomial Regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
'''
NOTE: in the 48 line we take poly_reg.fit_transform(x) instead of x_poly because x_poly
is already fitted to x so its not generalized for any x and hence we use it'''


'''
we can prepare x_grid for more continuous curve
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.title('truth or bluff(Polynomial Regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()'''

#predicting a new result via Linear Regression
lin_reg.predict([[6.5]])

#predicting a new result via Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]])) 




















