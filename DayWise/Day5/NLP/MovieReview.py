# -*- coding: utf-8 -*-
"""
@author: TSE
"""

import numpy as np
import pandas as pd

df = pd.read_csv('moviereviews.tsv', sep='\t')

#Detect & remove NaN values:
# Check for the existence of NaN values in a cell:
df.isnull().sum()

df.dropna(inplace=True)

#Detect & remove empty strings

blanks = []
for i,lb,rv in df.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)

print(len(blanks), 'blanks: ', blanks)

df.drop(blanks, inplace=True)

from sklearn.model_selection import train_test_split
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics

# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
text_clf_lsvc.fit(X_train, y_train)
predictions = text_clf_lsvc.predict(X_test)
cm=metrics.confusion_matrix(y_test,predictions)
acc=metrics.accuracy_score(y_test,predictions)


#TfidVectorizer(stop_words='english')

stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']

# RUN THIS CELL TO ADD STOPWORDS TO THE LINEAR SVC PIPELINE:
text_clf_lsvc2 = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)),
                     ('clf', LinearSVC())])
text_clf_lsvc2.fit(X_train, y_train)

predictions = text_clf_lsvc2.predict(X_test)
cm2=metrics.confusion_matrix(y_test,predictions)
acc2=metrics.accuracy_score(y_test,predictions)



