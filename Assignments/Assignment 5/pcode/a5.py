from __future__ import print_function

import tensorflow as tf 
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#loading the dataset

x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

print(x_train.shape)

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test, 10)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

print(x_test.shape)

train_size = 6000

X_train = np.zeros([train_size,32,32,1])
for i in range(train_size):
	im = Image.fromarray(x_train[i].reshape([28,28]))
	im = im.resize([32,32],resample = Image.BILINEAR)
	im = np.array(im)
	X_train[i,] = im.reshape([32,32,1])
x_train = X_train

'''
X_test = np.zeros([10000,32,32,1])
for i in range(10000):
	im = Image.fromarray(x_test[i].reshape([28,28]))
	im = im.resize([32,32],resample = Image.BOX)
	im = np.array(im)
	X_test[i,] = im.reshape([32,32,1])
x_test = X_test
'''


x_train = x_train/255;
x_test = x_test/255;
y_train = y_train[0:train_size]


input_shape = 32
batchsize = 100
learning_rate = 0.01
momentum = 0.8


model = Sequential()
#create a sequential model
model.add( Conv2D( 64,(3,3),padding = 'same',activation='relu',input_shape=(input_shape,input_shape,1)))
model.add(MaxPooling2D(padding = 'same',pool_size=(2,2)))
model.add( Conv2D( 128,(3,3),padding = 'same',activation='relu'))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2)))
model.add( Conv2D( 256,(3,3),padding = 'same',activation='relu'))
model.add( Conv2D( 256,(3,3),activation='relu'))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2)))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu'))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu'))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2)))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu' ))
model.add( Conv2D( 512,(3,3),padding = 'same',activation='relu' ))
model.add(MaxPooling2D(padding = 'same',pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))

model.add(Dense(10, activation='softmax'))
sgd = SGD(decay=1e-6, momentum=momentum, nesterov=True, lr = learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])


model.fit(x_train, y_train, batch_size=batchsize, epochs=10)
score = model.evaluate(x_test, y_test, verbose = 0)

print(score[1])