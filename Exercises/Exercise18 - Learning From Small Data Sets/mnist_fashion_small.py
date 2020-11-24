import tensorflow as tf
import numpy as np
import os
import distutils

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# add empty color dimension and scale to [0,1]
x_train = np.expand_dims(x_train, -1)/255
x_test = np.expand_dims(x_test, -1)/255

n_train = 1000
np.random.seed(2020)
ind = np.random.permutation(x_train.shape[0])
x_unlabeled =  x_train[ind[n_train:]]

x_train =  x_train[ind[:n_train]]
y_train = y_train[ind[:n_train]]

# Convert y to one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


