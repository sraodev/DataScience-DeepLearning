# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:20:17 2019

@author: TSE
"""

import pickle
modelCNN=pickle.load(open(r'CNNFullmodel.pkl','rb'))

import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'E:\catpd.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0).astype("float")

result = modelCNN.predict(test_image)



if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


