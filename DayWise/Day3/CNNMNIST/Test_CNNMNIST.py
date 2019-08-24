# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 20:00:36 2019

@author: TSE
"""
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D     # adding convolution layer for 2d images
from keras.layers import MaxPooling2D     # Adding Pooling Layers
from keras.layers import Flatten          
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
                            # num_of_filters/feature_map
                            # num_of_rows in feature detecter table
                            # num_of_rows in feature detecter table
classifier.add(Convolution2D(28, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


classifier.fit(x=x_train,y=y_train, epochs=10)


classifier.evaluate(x_test, y_test)

image_index = 3300
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = classifier.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())




