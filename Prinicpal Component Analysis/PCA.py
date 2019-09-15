# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:21:57 2019

@author: PALLAVI
"""
'''we will apply PCA to out model with logistic regression technique'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Wine.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,13].values

#splitting into test and training data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,shuffle=True)

'''feature scaling must be applied when applying PCA or LDA'''
#feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


'''we will apply PCA before applying our algorithm of regression'''
from sklearn.decomposition import PCA
pca=PCA(n_components=None)

'''as we need maximum of two components in our PCA model to finally predict the dataset but we will write None
becuase we do not know how much variance these two will produce and 
 we need to make sure that the two first principle components that explain the most variance do not 
explain it to low variance
because thn we are going to create a vector called explained variance and we will see that cumulative variance
is explained by all the components.np.array()

#fitting logistic regression to the training data set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
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
plt.title('LogisticRegression(Training Set)')
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
plt.title('LogisticRegression(test Set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()