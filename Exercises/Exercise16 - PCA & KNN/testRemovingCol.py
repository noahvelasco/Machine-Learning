# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:43:44 2020

@author: npizz
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import time
import math


if __name__ == "__main__":

    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise16\\'


    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    current_size = X.shape[1]
    i = 0
    while i < current_size:
        if np.var(X[:, i]) == 0:
            X = np.delete(X, i, axis = 1)
            current_size -= 1
        i += 1
        
    
    print(X_train.shape)
    print(X_test.shape)