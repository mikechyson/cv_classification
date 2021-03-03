#!/usr/bin/env python3
"""
@project: cv_classification
@file: LeNet
@author: mike
@time: 2021/3/3
 
@function:
"""
from keras.models import Sequential
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense


def LeNet(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)

    if K.image_data_format() == 'channels_first':
        input_shape = (depth, height, width)

    model.add(Conv2D(20, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model
