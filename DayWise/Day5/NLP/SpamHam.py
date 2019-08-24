

import numpy as np
import pandas as pd

#Load a dataset
df = pd.read_csv('smsspamcollection.tsv', sep='\t')

#Check for missing values:
df.isnull().sum()

#Split the data into train & test sets:
from sklearn.model_selection import train_test_split

X = df['message']  
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#Scikit-learn's CountVectorizer
#Text preprocessing, tokenizing and the ability to filter out 
#stopwords are all included in CountVectorizer, which builds a 
#dictionary of features and transforms documents to feature vectors.
"""
# below is a 2 step process VountVectorizer and Tfidf

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).toarray()

#Alternately we can do both steps in one 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

#Train a Classifier
#an SVM classifier that's similar to SVC, called LinearSVC. 
#LinearSVC handles sparse input better, 
#and scales well to large numbers of samples.

from sklearn.svm import LinearSVC
clf = LinearSVC()

clf.fit(X_train_tfidf,y_train)

"""

#Build a Pipeline
#scikit-learn offers a Pipeline class that behaves like a compound classifier.

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC()),])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)

#Test the classifier and display results
# Form a prediction set
predictions = text_clf.predict(X_test)

# Report the confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,predictions)

# overall accuracy
acc=metrics.accuracy_score(y_test,predictions)

#
predictions = text_clf.predict(["hello there"])
print(predictions)
#









