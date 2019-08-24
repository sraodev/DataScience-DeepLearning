
# =============================================================================
# implement the ML Model
# =============================================================================

#### Importing Libraries ####


import pandas as pd
import numpy as np

dataset = pd.read_csv('clean_churn_data.csv')

## Data Preparation

y=dataset.iloc[:,1].values  

dataset=dataset.drop(["user","churn"],axis=1)

# =============================================================================
# Step1 Check for number of unique values
# if numOfUnqVal > 2 , create dummy variables
# =============================================================================

print(len(dataset.housing.unique()))
print(len(dataset.zodiac_sign.unique()))
print(len(dataset.payment_type.unique()))

dataset =pd.get_dummies(dataset,drop_first=True)

X= dataset.iloc[:,:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Applying PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 35)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_
#
## 35 components 67%


#### Model Building ####

# Fitting Model to the Training Set
from sklearn.svm import SVC
classifier = SVC(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)














# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std()))






