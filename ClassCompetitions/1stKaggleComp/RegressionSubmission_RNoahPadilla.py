# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 01:02:46 2020

@author: R Noah Padilla

https://www.kaggle.com/c/cs-43615361-in-class-competition

> Goal <
    The goal of this competition is to predict the flux in next 30 minutes.
    Have fun :)

> Evaluation Metric <
    The evaluation metric for this competition is Mean Square Error (MSE)

> Submission Format <
    Submissions to the Kaggle leaderboard must take a specific form. 
    Submissions should be a csv with two columns, "ID" and "Prediction" and 
    55822 rows corresponding to the test observations. Including the header row, 
    there will be 55823 rows in the submission file. The "ID" is a unique numeric 
    identifier from "1", "2", ... to "55822". The "Prediction" is your predicted flux 
    for the test set.
    
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import time
import math

#Responsible for importing and extracting data into useful layout for data processing - used for x_test and x_train data
def import_xData():
    print("TODO")

#Responsible for importing extracting data into useful layout for data processing - used for y_train
def import_yData():
    print("TODO")

#Responsible for exporting results into a csv file - used for creating y_test
def export_yData():
    print("TODO")


if __name__ == "__main__":
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\ClassCompetitions\\1stKaggleComp'

    #X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    #y = np.load(data_path+'mnist_y.npy')
    
    #X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    