# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:11:30 2019

@author: PALLAVI
"""

'''
sometimes we do have problem s due to random selection in initial mean points
because this may lead to bad clusture formationand this is known as RANDOM INITIALIZATION TRAP
to solve this problem we use kmeans++ which happens by default in the background in python or R
'''
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#more the spending score more the spending rate
#we are required to form clustres based on the annual income and spending rate

#importing dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#using elbow method to find optimal value of k
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) #kmeans.inertia_ calculates wcss
plt.plot(range(1,11,1),wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('wcss values')
plt.show()

#it gives us value of k=5
#now appyling k means to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(x)

#visualizing it
#note that x[4][1] is same as x[4,1]
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,edgecolors='black',c='red',label='careful')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,edgecolors='black',c='blue',label='standard')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,edgecolors='black',c='green',label='target')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,edgecolors='black',c='cyan',label='careless')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,edgecolors='black',c='pink',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='magenta',edgecolors='black',label='centroids')
plt.title('cluster of clients')
plt.xlabel('Annual Income')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()