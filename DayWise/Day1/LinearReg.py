# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:43:33 2019

@author: TSE
"""

# =============================================================================
# import necessary packages
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
#  import data
# =============================================================================

dataset = pd.read_csv("01HR_Data.csv")

# =============================================================================
# Separate Features X independent, y dependent
# =============================================================================

X = dataset.iloc[:,[0]].values  # X 2d always, numpy arrays, tensors
y = dataset.iloc[:,1].values   # y 1d  always, numpy arrays, tensors

# =============================================================================
# Split data into training & testing
# 80-20/ 70-30  / 75-35
# =============================================================================
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3)

# =============================================================================
# Train the model
# =============================================================================

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt

plt.plot(y_pred,c="Red")
plt.plot(y_test,c="Green")
plt.show()















