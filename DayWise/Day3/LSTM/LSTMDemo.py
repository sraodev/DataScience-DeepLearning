# -*- coding: utf-8 -*-
"""
Long Short term Memory

Predict stock price of Google
Predict the upward and downward trend in Google stock price

We will make a LSTM that will capture upward and downward 
trend of Google stock price

We will train LSTM from 5yrs of Google stock price 
i.e. begining of 2012 to the end of 2016 , based on the correlation
identified by the LSTM , we will try to predict the first month 
of 2017 Jan,2017, its not going to predict the stock price
but will try to predict upward or downward trend of Google stock price

We will try to predict the open of the Google stock price
that is the open of the financial year
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data PreProcessing

# import the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
# must be a 2d, slicing open stock price 
training_set = dataset_train.iloc[:,[1]].values  

# Feature Sclaing using Normalization as we have sigmoid activation function

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1)) #all the new scaled stock price will be between 0 & 1
training_set_scaled = sc.fit_transform(training_set)

# number of timesteps tells the RNN what to remember to predict next value
# Creating a data structure with 60 timesteps and 1 output
# 60 timesteps means that at each time t the RNN is going to look at 60 stock price before time t 
# based on these 60 timesteps it will predict the next output
# 60 timesteps from past the RNN tries to understand some correlation and based on its understanding it 
# is going to try to predict the stock price
# 60 comes from experiments and 60 previous timesteps corresponds to 3months
# So we trying to look 3 months before to trying to predict the next day
# 60 timesteps and 1 output at time t+1

# X_train which will be the inputs of the neural network
# so for each financial day X_train will contain 60 previous stock prices before that financial day
# y_train which will contain the output
# y_train will contain the stock price of next financial day

X_train=[]   # vectors for passing information
y_train=[]   # vectors for passing information

# as we are going to get 60 previous stock prices , we start from the
# 60th financial day of 2012
for i in range(60,1258):
#    Stock price from 0 to 59
    X_train.append(training_set_scaled[i-60:i,0])
#    Stock price of 60th day
    y_train.append(training_set_scaled[i,0])

# convert X_train & y_train into numpy array
X_train = np.array(X_train) # 2d
y_train = np.array(y_train) # 1d

# Reshaping the data and adding more dimentionality
# we are adding unit that is the number of predictors we can use 
# to predict what we want, which is stock price at time t+1
                                                         
#                               batch_size      timestep    open price/number of indicators
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

# Initialising the RNN
regressor = Sequential() 

# Adding the first LSTM layer and some Dropout regularisation
# units number of LSTM cells / memory units / neurons in this LSTM layer
# return_sequsence True as you need to add one more LSTM layer
# input_shape is the timestep & indicators

regressor.add(LSTM(units=50, return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))  # dropout rate, drop 20% of neurons will be ignored in the LSTM layers during forward and back propogation
                             # 20% of 50 = 10 neurons will be droppedout
                             
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#return_sequences = False by default as we are not adding anymore LSTM layers

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#import pickle
#pickle.dump(regressor, open(r'E:\LSTMmodel.pkl','wb'))



# Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, [1]].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# stock price of prior 3 months 

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#modelLSTM = pickle.load(open(r'E:\LSTMmodel.pkl','rb'))
#predicted_stock_price_pkl = modelLSTM.predict(X_test)
#predicted_stock_price_pkl = sc.inverse_transform(predicted_stock_price_pkl)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

## Visualising the pickeled results
#plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
#plt.plot(predicted_stock_price_pkl, color = 'blue', label = 'Predicted Google Stock Price')
#plt.title('Google Stock Price Prediction')
#plt.xlabel('Time')
#plt.ylabel('Google Stock Price')
#plt.legend()
#plt.show()





