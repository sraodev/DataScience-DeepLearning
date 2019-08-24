# -*- coding: utf-8 -*-
"""
Manual ANN Implementation
Using Keras Functional API
"""

import numpy as np

# =============================================================================
# Meal Item Price Problem
# =============================================================================
#The true prices used by the cashier
p_fish = 150;p_chips = 50;p_ketchup = 100


#sample meal prices: generate data meal prices for 5 days.

np.random.seed(100)
portions = np.random.randint(low=1, high=10, size=3 )

X = [];y = [];days=10

for i in range(days):
    portions = np.random.randint(low=1, high=10, size=3 )
    price = p_fish * portions[0] + p_chips * portions[1] + p_ketchup * portions[2]    
    X.append(portions)
    y.append(price)
    
X = np.array(X)
y = np.array(y)

#Create a linear model
from keras.layers import Input, Dense 
from keras.models import Model
from keras.optimizers import SGD


price_guess = [np.array([[50],
                         [50],
                         [50]]) ]

#test_input = Input(shape=(3,2), dtype='float32')   
 
model_input = Input(shape=(3,), dtype='float32')
model_output = Dense(1, activation='linear', use_bias=False, 
                     name='LinearNeuron',
                     weights=price_guess)(model_input)
sgd = SGD(lr=0.01)
model = Model(model_input, model_output)  # Defining Layers using Keras Functional API

model.compile(loss="mean_squared_error", optimizer=sgd)

history = model.fit(X, y, batch_size=20, epochs=30,verbose=2)
l4  = history.history['loss']


model.get_layer('LinearNeuron').get_weights()
predWeights=model.get_layer('LinearNeuron').get_weights()

#p_fish = 150;p_chips = 50;p_ketchup = 100
# Actual output [8,8,1] => 8*150 =1200 , 8*50=400 , 1*100=100
#                       => 1700 /-
price_input = np.array([[8,8,1]]) 
model.predict(price_input)






