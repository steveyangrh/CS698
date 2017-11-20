import tensorflow as tf 
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#loading the dataset

model = Sequential()
#create a sequential model