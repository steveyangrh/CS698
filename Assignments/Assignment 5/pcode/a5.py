from __future__ import print_function

import tensorflow as tf 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#loading the dataset

model = Sequential()
	#create a sequential model

input_shape = 32

model.add( Conv2D( 64,(3,3),padding = 'same',activation='relu',input_shape=(input_shape,input_shape,1)))
model.add(MaxPooling2D(padding = 'same',pool_size=(2,2),strides=None))
model.add( Conv2D( 128,(3,3),padding = 'same',activation='relu'))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2)))
model.add( Conv2D( 256,(3,3),padding = 'same',activation='relu'))
model.add( Conv2D( 256,(3,3),activation='relu'))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2),strides=None))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu'))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu'))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2),strides=None))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu' ))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu' ))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2),strides=None))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Flatten())

model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

