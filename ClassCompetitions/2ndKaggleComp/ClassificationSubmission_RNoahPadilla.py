# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 00:53:47 2020

@author: R Noah Padilla

Evaluation: 
    The evaluation metric for this competition is Mean F1-Score. The F1 score, 
    commonly used in information retrieval, measures accuracy using the statistics 
    precision p and recall r. Precision is the ratio of true positives (tp) to all 
    predicted positives (tp + fp). Recall is the ratio of true positives to all 
    actual positives (tp + fn). The F1 score is given by:
    
            > F1=2p⋅rp+r  where  p=tptp+fp,  r=tptp+fn
    
    
    The F1 metric weights recall and precision equally, and a good retrieval algorithm 
    will maximize both precision and recall simultaneously. Thus, moderately good 
    performance on both will be favored over extremely good performance on one and poor 
    performance on the other.

Submission Format:
    Submissions to the Kaggle leaderboard must take a specific form. 
    Submissions should be a csv with two columns, "ID" and "Prediction" and 
    55822 rows corresponding to the test observations. Including the header row,
    there will be 55823 rows in the submission file. The "ID" is a unique numeric 
    identifier from "1", "2", ... to "55822". The "Prediction" is your predicted 
    flux for the test set.
    
Data Description:
    Each row of x_train (and x_test) represents flares for 1 hour and y_train 
    represents if there is any flare in next 30 minutes.
    x-val and corresponding y_val is also provided to check the performance of the model.

"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from utils import *
import time

if __name__ == "__main__":
    
    x_train = pd.read_csv('.\\x_train.csv',header=None).to_numpy()
    y_train = pd.read_csv('.\\y_train.csv',header=None).to_numpy()
    x_practice_test = pd.read_csv('.\\x_val.csv',header=None).to_numpy()
    y_practice_test = pd.read_csv('.\\y_val.csv',header=None).to_numpy()
    x_test = pd.read_csv('.\\x_test.csv',header=None).to_numpy()
    
    
    '''
    class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', 
                                               *, solver='adam', alpha=0.0001, batch_size='auto', 
                                               learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                                               max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
                                               warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                                               early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                                               beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    '''
    
    
    model = MLPClassifier(hidden_layer_sizes=(100, ) , alpha=1e-5 , batch_size=100, learning_rate='adaptive' , early_stopping=True, verbose=True)
    
    #Fit the training data
    start = time.time()
    model.fit(x_train, y_train.ravel())
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} secs'.format(elapsed_time))  
    
    #Test on test set
    start = time.time()
    pred = model.predict(x_practice_test)
    elapsed_time = time.time()-start
    print('Elapsed time testing practice test set {0:.6f} secs'.format(elapsed_time))
    print('Accuracy on practice test set: {0:.6f}'.format(accuracy(y_practice_test,pred)))
    
    '''
    start = time.time()       
    pred = model.predict(x_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))   
    '''
    
    
    