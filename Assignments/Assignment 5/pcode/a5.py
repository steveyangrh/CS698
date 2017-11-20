import tensorflow as tf 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#loading the dataset

model = Sequential()
#create a sequential model