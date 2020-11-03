# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 01:17:37 2020

@author: R Noah Padilla


Solar Particle Dataset
"""

import numpy as np
from utils import *
import math 
import time
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":
    print('RANDOM FOREST REGRESSOR')

    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise15\\'

    
    X = np.load(data_path+'particles_X.npy')
    y = np.load(data_path+'particles_y.npy')
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    '''
    class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, 
                                                 criterion='mse', max_depth=None,
                                                 min_samples_split=2, min_samples_leaf=1,
                                                 min_weight_fraction_leaf=0.0, 
                                                 max_features='auto', max_leaf_nodes=None, 
                                                 min_impurity_decrease=0.0, 
                                                 min_impurity_split=None, 
                                                 bootstrap=True, oob_score=False,
                                                 n_jobs=None, random_state=None, 
                                                 verbose=0, warm_start=False, 
                                                 ccp_alpha=0.0, max_samples=None)
    n_estimators - The number of trees in the forest.
    '''
    estimators = [] #Holds the number of trees in a forest(indicated by index)
    trainTime=[]    #Holds all the train times
    testTime=[]     #Holds all the test times
    all_mse=[]      #Holds all the mses's
    
    #tested with small vals prior so 10-70 is a reasable threshold
    for est in range(10,70,10):
        print("> Num of estimators: ", est)
        estimators.append(est)
        
        model = RandomForestRegressor(n_estimators=est,random_state=0)
        
        start = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time()-start
        print('Elapsed time training {:.6f} secs'.format(elapsed_time))
        trainTime.append(elapsed_time)
        
        start = time.time()
        pred = model.predict(X_test)
        elapsed_time = time.time()-start
        print('Elapsed time testing {:.6f} secs'.format(elapsed_time))
        testTime.append(elapsed_time)
        
        print('mse: {:.6f}'.format(mse(pred, y_test)))
        all_mse.append(mse(pred, y_test))
        
        print()
        
    print("\n**-- LOWEST MSE stats --**")
    print("Num of estimators: ", estimators[np.argmin(all_mse)])
    print("Train Time: ", trainTime[np.argmin(all_mse)])
    print("Test Time: ", testTime[np.argmin(all_mse)])
    print("MSE: ", np.min(all_mse))
    
'''
--- RandomForestRegressor(n_estimators=1,random_state=0) ---
RANDOM FOREST REGRESSOR
Elapsed time training 13.133794 secs
Elapsed time testing 0.200902 secs
Ensemble mse: 0.076286

--- RandomForestRegressor(n_estimators=10,random_state=0) ---
RANDOM FOREST REGRESSOR
Elapsed time training 131.810516 secs
Elapsed time testing 1.581896 secs
Ensemble mse: 0.042233

--- RandomForestRegressor(n_estimators=1,random_state=0) ---
'''