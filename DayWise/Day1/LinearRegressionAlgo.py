# -*- coding: utf-8 -*-
"""
Linear Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("01HR_Data.csv")

X = dataset.iloc[:,[0]].values   # 2d
y = dataset.iloc[:,1].values     # 1d

# =============================================================================
#  80% for training   20% testing
#  70-20   75-25 
# random split ,  seed of randomness   random_state 
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

# =============================================================================
# Implement Linear Regression
# =============================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# =============================================================================
# Train the machine , i.e. it learns patterns , claculates slope & intercept
# y = b0 + b1x1
# =============================================================================

regressor.fit(X_train,y_train)


#regressor.predict([[5]])

y_pred = regressor.predict(X_test)

# =============================================================================
# Compare y_pred(predicted) with y_test(actual)
# =============================================================================

plt.plot(y_pred,c="Red")
plt.plot(y_test,c="Green")
plt.show()

















