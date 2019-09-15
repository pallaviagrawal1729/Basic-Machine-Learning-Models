# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:50:02 2019

@author: PALLAVI
"""

'''
Natural Language Processing (or NLP) is applying Machine Learning models to text and language. Teaching machines to understand 
what is said in spoken and written word is the focus of Natural Language Processing. Whenever you dictate something 
into your iPhone / Android device that is then converted to text, that’s an NLP algorithm in action.

You can also use NLP on a text review to predict if the review is a good one or a bad one. You can use NLP on an 
article to predict some categories of the articles you are trying to segment. You can use NLP on a book to predict 
the genre of the book. And it can go further, you can use NLP to build a machine translator or a speech recognition 
system, and in that last example you use classification algorithms to classify language. Speaking of classification 
algorithms, most of NLP algorithms are classification models, and they include Logistic Regression, Naive Bayes, CART 
which is a model based on decision trees, Maximum Entropy again related to Decision Trees, Hidden Markov Models which 
are models based on Markov processes.

A very well-known model in NLP is the Bag of Words model. It is a model used to preprocess the texts to classify 
before fitting the classification algorithms on the observations containing the texts.

In this part, you will understand and learn how to:

    Clean texts to prepare them for the Machine Learning models,
    Create a Bag of Words model,
    Apply Machine Learning models onto this Bag of Worlds model.
'''
'''from wikipedia:
    This list (or vector) representation does not preserve the order of the words in the original sentences. 
    This is just the main feature of the Bag-of-words model. '''
#predictive analysis on text 
#we make a model to predict whether the review is positive or negative
#TSV-Tab Separated Values
#CSV-Comma Separated Value
#in our file 1 means positive reiview and 0 means negative review
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
consider a review and liked as :
    "The food, amazing.",1 now the food is considered review and as separated by comma amazing is considered as liked
    and the 1 will be treated as review for the next  
so to eliminate such problems we use tsv
'''

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#quoting=3 is used to ignore doble quotes.
'''now our motive is to clear the words that will be useless in bag of words model
like 1.  'the' , 'is' etc, because all these are grammitical editing and play no role.
2. we will also get rid of punctuations
3. we will also apply stimming it will make loved or loving as love...this will reduce number of words and will leave only important 
words having meaningful result
4. we will also make all capital letters to small so that same words do not fall into bag
and then we finally apply tokenization process so that it finally splits the lines into different valid words
'''

'''also note that we use dataset[column][row] to access particular block'''
#FOR UNDERSTANDING:
'''
#[^a-zA-Z] not to be removed (^ is used for not removal)
#space is used to replace whatever we remove so that words do not combine and give a different meaning
#dataset['Review'][0] is the one on which we are applying this.
import re
#review=[]
#for i in range(0,1000):
#    review.append(re.sub('[^a-zA-Z]',' ',dataset['Review'][i]))
review=re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
#for all lower cases:
review=review.lower()
#to get rid of irrelevant words


#for this we will use stopwords library which included all the irrelevant words that should be removed before predicting 
#the review. Now this library works on list of elements but by above processing we get a line so firstly we need to convert
#it into list of words.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#for stemming
from nltk.stem.porter import PorterStemmer
#creating object for stemming
ps=PorterStemmer()
review=review.split()
#stopwords.words('english') checks for word only in english language because stopwords include for many lang.
#we use set function when we have a long article or many words because it increases the speed as searching is faster 
#in set thn the list using some algorithms 
#ps.stem(word) stems ths word like 'loved' or 'loving' to love
review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
#now we want to make it back into string
review=' '.join(review)
'''
#MY TRIAL 
'''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
review=[]
for i in range(0,1000):
    review.append(re.sub('[^a-zA-Z]',' ',dataset['Review'][i]))
    review[i]=review[i].lower()
    review[i]=review[i].split()
    review[i]=[ps.stem(word) for word in review[i] if word not in set(stopwords.words('english'))]
    review[i]=' '.join(review[i])
'''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#creang bags of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#to decrease the sparcity we can either user max_features or dimensionality reduction

#we can use any classification algo now but generally we use decision tree or naive bayes and here we use second one

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




