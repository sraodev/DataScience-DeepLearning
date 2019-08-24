# Natural Language Processing is an area of Artificial Intelligence

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
# a tab separated file as , "" will be a part of the content and hence we dont prefer csv file
# quoting = 3 is the code to ignore the double quotes

dataset = pd.read_csv('06Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re     # librarie for cleaning data
import nltk   # library for NLP
nltk.download('stopwords')  # stopwords pakage is a preexisting list of stopwords  (the, is , this, there...)
from nltk.corpus import stopwords   
from nltk.stem.porter import PorterStemmer    # class for stemming

corpus = []         # variable corpus of type list is a collection of text, so this variable will contain the cleaned 1000 reviews 

stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])         # cleaned reviews in variable review, this step will only keep the letters from A-z in the review and  remove  the numbers, puntuation part, exclanmations,question marks
                    # [^a-zA-Z] indicates what we dont want to remove 
                    # Replace the removed character by space 
    review = review.lower()
                    # convert the reviews in lower case
    review = review.split()
                    # split the string into words
                    
    ps = PorterStemmer()
                    
#    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [ps.stem(word) for word in review if not word in stopset]
                    #steming is keepig only the parent word love is root of loveable,loved,lovely
    review = ' '.join(review)
                    # join the words back to make a sentence
    corpus.append(review)
                    # appending the cleaned reviews to corpus 
                

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)                      # max_features = 1500 just retaining 1500 relevant words
X = cv.fit_transform(corpus).toarray()    # toarray makes it a matrix
y = dataset.iloc[:, 1].values             # dependent variable for column Liked in dataset

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc= accuracy_score(y_test,y_pred)


# accuracy (55+91)/200 in X_test 
# 55 correct predictions of negative reviews cm[0][0]
# 12 incorrect predictions of negative reviews cm [1][0]
# 42 incorrect predictions of positive reviews cm[0][1]
# 91 correct for positiove reviews  cm[1][1]
