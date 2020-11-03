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
    for est in range(50,350,50):
        print("> Num of estimators: ", est)
        estimators.append(est)
        
        model = RandomForestRegressor(max_depth = 12,n_estimators=est,random_state=0)
        
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
RANDOM FOREST REGRESSOR - Only modifying estimators
> Num of estimators:  10
Elapsed time training 132.753099 secs
Elapsed time testing 1.548245 secs
mse: 0.042233

> Num of estimators:  20
Elapsed time training 263.488964 secs
Elapsed time testing 3.179020 secs
mse: 0.040254

> Num of estimators:  30
Elapsed time training 385.591375 secs
Elapsed time testing 4.292018 secs
mse: 0.039666

> Num of estimators:  40
Elapsed time training 512.790967 secs
Elapsed time testing 5.907152 secs
mse: 0.039384

> Num of estimators:  50
Elapsed time training 639.141176 secs
Elapsed time testing 7.237906 secs
mse: 0.039186

> Num of estimators:  60
Elapsed time training 742.210262 secs
Elapsed time testing 8.695497 secs
mse: 0.039052


**-- LOWEST MSE stats --**
Num of estimators:  60
Train Time:  742.210262298584
Test Time:  8.695496797561646
MSE:  0.039052008375852744

> Num of estimators:  100
Elapsed time training 1271.615759 secs
Elapsed time testing 67.963341 secs
mse: 0.038782

> Num of estimators:  150
Elapsed time training 1977.593288 secs
Elapsed time testing 1412.404979 secs
mse: 0.038656
-------------------------------------------------------------------------------

RANDOM FOREST REGRESSOR - Only modifying estimators and max_depth=12
> Num of estimators:  50
Elapsed time training 276.178408 secs
Elapsed time testing 0.961541 secs
mse: 0.037777

> Num of estimators:  100
Elapsed time training 571.345402 secs
Elapsed time testing 1.933005 secs
mse: 0.037768

> Num of estimators:  150
Elapsed time training 855.273388 secs
Elapsed time testing 2.904613 secs
mse: 0.037766

> Num of estimators:  200
Elapsed time training 1125.475703 secs
Elapsed time testing 3.815182 secs
mse: 0.037763

> Num of estimators:  250
Elapsed time training 1409.710879 secs
Elapsed time testing 4.409895 secs
mse: 0.037763

> Num of estimators:  300
Elapsed time training 1669.788249 secs
Elapsed time testing 5.799232 secs
mse: 0.037762


**-- LOWEST MSE stats --**
Num of estimators:  300
Train Time:  1669.7882494926453
Test Time:  5.799232482910156
MSE:  0.03776189518489129

'''