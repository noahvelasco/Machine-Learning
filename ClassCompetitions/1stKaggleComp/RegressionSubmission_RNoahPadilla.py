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

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import numpy as np
import pandas as pd
from utils import *
import time

if __name__ == "__main__":
    
    #Over 250,000 train data samples and 60,000 test data samples
    #X_train = pd.read_csv('.\\x_train.csv',header=None).to_numpy()
    #y_train = pd.read_csv('.\\y_train.csv',header=None).to_numpy()
    #X_test = pd.read_csv('.\\x_test.csv',header=None).to_numpy()
    
    X_train = np.load('x_train.npy').astype(np.float32)
    X_test = np.load('x_test.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    
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
    
    #model = RandomForestRegressor(max_depth = 12,n_estimators=300,verbose=True , random_state=0)
    
    '''
    MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', 
                 alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                 warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, 
                 max_fun=15000)
    '''
    '''
    class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, 
                          C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    
    '''
    #mlp regressor best so far with a score of .01212
    #model = MLPRegressor(solver='adam', activation='relu' ,  alpha=1e-8,learning_rate_init=0.001,batch_size = 100 ,learning_rate='adaptive', hidden_layer_sizes=(27,100),early_stopping = True, verbose=True, random_state=1)
    
    #support vector machine
    model = svm.SVR(kernel = 'poly', verbose=True)
    
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed time training {:.6f} secs'.format(elapsed_time))
    
    start = time.time()
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed time testing {:.6f} secs'.format(elapsed_time))
    
    '''
    > Submission Format <
    Submissions to the Kaggle leaderboard must take a specific form. 
    Submissions should be a csv with two columns, "ID" and "Prediction" and 
    55822 rows corresponding to the test observations. Including the header row, 
    there will be 55823 rows in the submission file. The "ID" is a unique numeric 
    identifier from "1", "2", ... to "55822". The "Prediction" is your predicted flux 
    for the test set.
    
    VVVV  https://datatofish.com/export-dataframe-to-csv/ VVVV
    
    cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22000,25000,27000,35000]
        }

    df = pd.DataFrame(cars, columns= ['Brand', 'Price'])

    '''
    
    #Export the predictions
    print("NOW EXPORTING...")
    
    
    #Set up the layout
    int_indices = np.arange(1,len(pred)+1)
    str_indices = []
    for i in range(len(int_indices)):
        str_indices.append(str(int_indices[i]))
    
    df = pd.DataFrame({"ID": str_indices,
                   "Prediction": pred })
    df.to_csv('Predictions_SVR.csv',index=False)
    
    print("...FINISHED EXPORTING")
    '''

$$$$$$$$$$$$$$$$$$$$$$$$$$$$ USING RANDOM FOREST REGRESSOR - kaggle score initiated at 0.01249 

Elapsed time training 2881.168830 secs
Elapsed time testing 3.078863 secs
NOW EXPORTING...
...FINISHED EXPORTING

Elapsed time training 2760.919721 secs
Elapsed time testing 2.866538 secs
NOW EXPORTING...
...FINISHED EXPORTING

$$$$$$$$$$$$$$$$$$$$$$$$$$$$ USING MLP REGRESSOR - kaggles score improved to 0.01235
Iteration 1, loss = 0.00735878
Iteration 2, loss = 0.00654856
Iteration 3, loss = 0.00632938
Iteration 4, loss = 0.00618745
Iteration 5, loss = 0.00615222
Iteration 6, loss = 0.00612697
Iteration 7, loss = 0.00608946
Iteration 8, loss = 0.00605923
Iteration 9, loss = 0.00606257
Iteration 10, loss = 0.00605809
Iteration 11, loss = 0.00604016
Iteration 12, loss = 0.00604250
Iteration 13, loss = 0.00603701
Iteration 14, loss = 0.00602677
Iteration 15, loss = 0.00600878
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
Elapsed time training 46.632797 secs
Elapsed time testing 0.155923 secs
NOW EXPORTING...
...FINISHED EXPORTING

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ USING MLP REGRESSOR - kaggles score improved to 0.01235
    model = MLPRegressor(solver='adam', alpha=1e-5,learning_rate_init=0.001,batch_size = 100 ,learning_rate='adaptive',momentum=0.95,  hidden_layer_sizes=(300,100), verbose=True, random_state=1)

Iteration 1, loss = 0.00699709
Iteration 2, loss = 0.00628216
Iteration 3, loss = 0.00620250
Iteration 4, loss = 0.00614819
Iteration 5, loss = 0.00609570
Iteration 6, loss = 0.00607099
Iteration 7, loss = 0.00606345
Iteration 8, loss = 0.00603876
Iteration 9, loss = 0.00602648
Iteration 10, loss = 0.00601621
Iteration 11, loss = 0.00599386
Iteration 12, loss = 0.00599050
Iteration 13, loss = 0.00599000
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
Elapsed time training 76.720385 secs
Elapsed time testing 0.271871 secs
NOW EXPORTING...
...FINISHED EXPORTING

solver='adam', alpha=1e-5,learning_rate_init=0.001,batch_size = 100 ,learning_rate='adaptive',momentum=0.95,  hidden_layer_sizes=(300,150), verbose=True, random_state=1
Iteration 1, loss = 0.00719960
Iteration 2, loss = 0.00630550
Iteration 3, loss = 0.00619825
Iteration 4, loss = 0.00613875
Iteration 5, loss = 0.00608846
Iteration 6, loss = 0.00609190
Iteration 7, loss = 0.00605455
Iteration 8, loss = 0.00602577
Iteration 9, loss = 0.00602562
Iteration 10, loss = 0.00599842
Iteration 11, loss = 0.00599448
Iteration 12, loss = 0.00598325
Iteration 13, loss = 0.00598437
Iteration 14, loss = 0.00595921
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
Elapsed time training 102.587042 secs
Elapsed time testing 0.313851 secs
NOW EXPORTING...
...FINISHED EXPORTING

model = MLPRegressor(solver='adam', activation='relu' ,  alpha=1e-7,learning_rate_init=0.001,batch_size = 100 ,learning_rate='adaptive',momentum=0.95,  hidden_layer_sizes=(300,150), verbose=True, random_state=1)
Iteration 1, loss = 0.00719474
Iteration 2, loss = 0.00628540
Iteration 3, loss = 0.00617857
Iteration 4, loss = 0.00612013
Iteration 5, loss = 0.00607599
Iteration 6, loss = 0.00606609
Iteration 7, loss = 0.00603671
Iteration 8, loss = 0.00601641
Iteration 9, loss = 0.00601731
Iteration 10, loss = 0.00598969
Iteration 11, loss = 0.00599364
Iteration 12, loss = 0.00597875
Iteration 13, loss = 0.00597645
Iteration 14, loss = 0.00595430
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
Elapsed time training 98.094186 secs
Elapsed time testing 0.315849 secs
NOW EXPORTING...
...FINISHED EXPORTING

model = MLPRegressor(solver='adam', activation='relu' ,  alpha=1e-8,learning_rate_init=0.001,batch_size = 100 ,learning_rate='adaptive', hidden_layer_sizes=(300,150), early_stopping = True, verbose=True, random_state=1)
Iteration 1, loss = 0.00724563
Validation score: 0.931185
Iteration 2, loss = 0.00628296
Validation score: 0.934124
Iteration 3, loss = 0.00620107
Validation score: 0.935715
Iteration 4, loss = 0.00613286
Validation score: 0.933888
Iteration 5, loss = 0.00612876
Validation score: 0.935384
Iteration 6, loss = 0.00608387
Validation score: 0.933135
Iteration 7, loss = 0.00603381
Validation score: 0.935899
Iteration 8, loss = 0.00602400
Validation score: 0.935333
Iteration 9, loss = 0.00602513
Validation score: 0.936542
Iteration 10, loss = 0.00601196
Validation score: 0.936399
Iteration 11, loss = 0.00601339
Validation score: 0.933785
Iteration 12, loss = 0.00599377
Validation score: 0.936118
Iteration 13, loss = 0.00599015
Validation score: 0.936743
Iteration 14, loss = 0.00598232
Validation score: 0.935408
Iteration 15, loss = 0.00597500
Validation score: 0.936375
Iteration 16, loss = 0.00595656
Validation score: 0.935717
Iteration 17, loss = 0.00595616
Validation score: 0.936711
Iteration 18, loss = 0.00595241
Validation score: 0.936441
Iteration 19, loss = 0.00594686
Validation score: 0.936951
Iteration 20, loss = 0.00594818
Validation score: 0.935381
Iteration 21, loss = 0.00593855
Validation score: 0.936236
Iteration 22, loss = 0.00594788
Validation score: 0.936376
Iteration 23, loss = 0.00592571
Validation score: 0.937040
Iteration 24, loss = 0.00593759
Validation score: 0.936036
Iteration 25, loss = 0.00592864
Validation score: 0.936829
Iteration 26, loss = 0.00592335
Validation score: 0.936944
Iteration 27, loss = 0.00592842
Validation score: 0.936246
Iteration 28, loss = 0.00592950
Validation score: 0.936603
Iteration 29, loss = 0.00592319
Validation score: 0.936618
Iteration 30, loss = 0.00591517
Validation score: 0.935950
Validation score did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
Elapsed time training 199.945797 secs
Elapsed time testing 0.279867 secs
NOW EXPORTING...
...FINISHED EXPORTING






    '''