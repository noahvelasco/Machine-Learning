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

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import pandas as pd

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
    
    model = RandomForestRegressor(max_depth = 12,n_estimators=300,random_state=0)
    
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
    df.to_csv('Predictions.csv',index=False)
    
    print("...FINISHED EXPORTING")
    '''

---- 1st submission ---- using Random Forest Regressor

Elapsed time training 2881.168830 secs
Elapsed time testing 3.078863 secs
NOW EXPORTING...
...FINISHED EXPORTING

Elapsed time training 2760.919721 secs
Elapsed time testing 2.866538 secs
NOW EXPORTING...
...FINISHED EXPORTING

    '''