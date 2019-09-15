# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:36:26 2019

@author: PALLAVI
"""

'''
according to naive bayes algorithm if P(A) is given then
P(B|A)=P(A|B)*P(B)/P(A)
suppose you have wrenches from different sets A and B and both produces defected wrenches 
to find how many defected wrenches are produced by each machine we will directly fin the probablity
but now if we are required to pick up a wrench from defected slot and predict to which machine it belongs we will
have to use naie bayes 'given that the piece is broken what is the probability that it came from machine 1  
'''

'''
now suppose we have a random point X with different features of age and salary and we wantt o classify it into 
walker or driver(a person who walks or drives a car)
so we will check P(Walks|X) v.s. P(Drives|X)
to find a particular value above if we put it into the formula we get
P(Walks|X)=P(X|Walks)*P(Walks)/P(X)
#1 P(Walks)= PRIOR PROBABILITY
           =people walking/total population
           =10/30
#2 P(X)= MARGINAL PROBABILITY (what is the likelihood of population having features X)
       = people in the circle have same features/total population
       = 4/30
#3 P(X|Walks) = LIKELIHOOD (given that a person is walking what is the probability that he also possess features X)
                (or how many among walkers possess feature X)
            = walkers in the circle have same feature as X/walkers among whole population
            =3/10
#4 P(Walks|X)= POSTERIOR PROBABILITY 
             = (3/10*10/30)/4/30=0.75
 similarly we calculate  P(Drives|X)=0.25
 comparing we can see that P(Walks|X) has more probability and hence the person walks given his fetures of age and 
 salary
             
'''
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
datasets=pd.read_csv('Social_Network_Ads.csv')
x=datasets.iloc[:,[2,3]].values
y=datasets.iloc[:,4].values

#splitting into test and training data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#fit the classifier  to the training data set
#create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#predicting the test set results
y_pred=classifier.predict(x_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#visualizing the training set results
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('NAIVE BAYES(Training Set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()

#visualizing the test set results
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('NAIVE BAYES(test Set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()