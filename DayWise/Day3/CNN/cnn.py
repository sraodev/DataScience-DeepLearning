# Convolutional Neural Network


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D     # adding convolution layer for 2d images
from keras.layers import MaxPooling2D     # Adding Pooling Layers
from keras.layers import Flatten          
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# =============================================================================
# #  Convolution Step
# =============================================================================
                            # num_of_filters/feature_map
                            # num_of_rows in feature detecter table
                            # num_of_rows in feature detecter table
# Adding a first convolutional layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#  Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional & Pooling layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

#Image Augmentation

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#Image Augmentation

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

training_set.class_indices

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'single_prediction\cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)

#(samples, width, height, color_depth).
test_image = np.expand_dims(test_image, axis = 0).astype("float")

result = classifier.predict(test_image)


if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


