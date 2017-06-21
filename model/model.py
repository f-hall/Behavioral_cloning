
# coding: utf-8

# I left the old comments, from the first submit in the code and added some, where they are neede
# with RESUBMIT: beforehand.

import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.normalization import BatchNormalization

# Import all images and steering angles from data.
# I used the 'left', 'center' and 'right' images from the provided udacity dataset
# Choosing steering angles y_left = y_center + 0.3 and y_right = y_center - 0.3 to get more data
# From the provided data I take only approx. 20% of the center images to get less data with steering angle equal to zero.
# Furthermore I used data that I produced with the simulator (that has only center images) for the more
# difficult curves on the track.

# In addition I flipped all images and associated steering angles to get more data.
# Resizing the pictures to 64x32 pix, normalizing and cutting the upper part of the pictures,
# so I get approx. 52.000 64x22 pix images.
# Did not need a python generator or gpu, which is nice.

# Shuffling x_array / y_array

#...............................................................................................

# RESUBMIT: Used a python generator instead of loading all data in memory. The training of the model
# is now double the time before, because you have to read from disk all the time. But it is advantages
# when you don't have enough memory to store all image data in it.

training_file = "/media/frank/Zusatz/CarND-P3/data_add/driving_log.csv"
with open (training_file) as csvfile:
    reader = csv.DictReader(csvfile)
    x_array = []
    y_array = []
    for row in reader:
        _, x = row['center'].split('IMG/')
        y = float(row['steering'])
        yl = y + 0.3
        yr = y - 0.3
        space = 0
        if not row['left'].isspace():
            _, l = row['left'].split('IMG/')
            _, r = row['right'].split('IMG/')
            space = 1
        rnd = np.random.uniform()
        if (rnd > 0.8) or (space == 0):
            x_array.append(x)
            y_array.append(y)
        if (space == 1):
            x_array.append(l)
            x_array.append(r)
            y_array.append(yl)
            y_array.append(yr)
            
x_array = np.asarray(x_array)
y_array = np.asarray(y_array)
x_array, y_array = shuffle(x_array, y_array)
x_train, x_val = np.split(x_array, [int(0.8*len(x_array))])
y_train, y_val = np.split(y_array, [int(0.8*len(y_array))])
print(x_array.shape)
print(y_array.shape)

def gen(x, y, batch_size):
    rows = x.shape[0]
    counter = None
    while 1:
        X_list = []
        Y_list = []
        for i in range(batch_size):
            if counter is None or counter >= rows:
                counter = 0
            X = x[counter]
            Y = y[counter]
            X = Image.open('/media/frank/Zusatz/CarND-P3/data_add/IMG/'+X)
            X = np.asarray(X)
            X = cv2.resize(X, None, fx=0.2, fy=0.2)
            X = X / 255
            X = X[10:32, 0:64]
            flip_rnd = np.random.uniform()
            if flip_rnd > 0.5:
                X = cv2.flip(X, 1)
                Y = -1 * Y
            X_list.append(X)
            Y_list.append(Y)
            counter = counter + 1
        X_list = np.asarray(X_list).reshape(-1, 22, 64, 3)
        Y_list = np.asarray(Y_list)
        yield X_list, Y_list
        
            


# Definition of my model used for learing, testing and using it in the simulation.
# It was just a ton of trial and error to get to this, but lastly I think that the
# chosen data and the preprocessing is much more important.

# model: Conv (8x2x2xsame) - AvgPool(2x2) - Dropout(0.25) - ELU activation
# - Conv(16x2x2xsame) - ELU activation - Conv(32x2x2xsame) - ELU activation 
# - Conv(48x2x2xsame) - ELU activation - Flatten - Dense(256) - Dropout(0.40)
# - ELU activation - Dense(128) - ELU activation - Dense(64) - Dropout(0.40)
# - ELU activation - Dense(32) - ELU activation - Dense(1)

# I have choosen mse and rmsprop. rmsprop seems to work better for me than 
# adam optimizer for different learning rates does. 
# Default lr (0.001)

#..............................................................................

# RESUBMIT: Used BatchNormalization before each ELU-Activationlayer.


def get_model():
    model = Sequential()
    model.add(Convolution2D(8, 2, 2, input_shape=(22, 64, 3)))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(16, 2, 2))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 2, 2))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(48, 2, 2))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.40))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(64))
    model.add(Dropout(0.40))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))
    
    model.compile(loss = 'mse', optimizer='rmsprop')
    return model


# Testing showed batch_size 256 is good for me.
# Validationsplit 20% of data to prevent overfitting.
# Used 15 epochs before, but 30 seems to be more stable and with
# the dropout it seems I'm far away from overfitting, so it should not be a problem.
# Maybe 25 or 20 epoch are enough, but I did not test this.
# For me it was exactly 1 min for each epoch, so 30 minutes for the whole thing.

#.............................................................................................

# RESUBMIT: Used model.fit_generator instead of model.fit

model = get_model()
model.fit_generator(gen(x_train, y_train, 256), samples_per_epoch=x_train[0], nb_epoch=30, validation_data=gen(x_val, y_val, 256), nb_val_samples=x_val[0])


# Saving the model and the weights


from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

# I used more testing cells in this notebook before but cutted them out before the
# last version for a better general view.





