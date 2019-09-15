# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:45:17 2019

@author: PALLAVI
"""

'''
to find optimal number of clusters that should be formed we are going to use a method
in this we take laregest distance line from dendogram and make our threshold line passing through it'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#using dendogram for finding optimal nuer of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('eucledian distances')
plt.show()

#fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

#visualizing clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,edgecolors='black',c='red',label='careful')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,edgecolors='black',c='blue',label='standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,edgecolors='black',c='green',label='target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,edgecolors='black',c='cyan',label='careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,edgecolors='black',c='pink',label='sensible')
plt.title('cluster of clients')
plt.xlabel('Annual Income')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()