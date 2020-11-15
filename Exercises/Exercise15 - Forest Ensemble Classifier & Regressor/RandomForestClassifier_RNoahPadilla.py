# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 01:16:58 2020

@author: R Noah Padilla


MNIST Dataset
"""

import numpy as np
from utils import *
import time
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    print('RANDOM FOREST CLASSIFIER')

    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise15\\'


    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)
    y = np.load(data_path+'mnist_y.npy')
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    '''
    class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', 
                                                  max_depth=None, min_samples_split=2, 
                                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                                  max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                                  bootstrap=True, oob_score=False, 
                                                  n_jobs=None, random_state=None, 
                                                  verbose=0, warm_start=False, 
                                                  class_weight=None, ccp_alpha=0.0, 
                                                  max_samples=None)
    
    n_estimators - The number of trees in the forest.
    '''
    
    estimators = [] #Holds the number of trees in a forest(indicated by index)
    trainTime=[]    #Holds all the train times
    testTime=[]     #Holds all the test times
    acc=[]          #Holds all the accuracies
    
    for est in range(50,350,50):
        print("> Num of estimators: ", est)
        estimators.append(est)
        
        model = RandomForestClassifier(n_estimators=est,random_state=0)
        
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
        
        print('Ensemble accuracy: {:.6f}'.format(accuracy(y_test,pred)))
        acc.append(accuracy(y_test,pred))
        
        print()

    print("\n**-- Best ACC stats --**")
    print("Num of estimators: ", estimators[np.argmax(acc)])
    print("Train Time: ", trainTime[np.argmax(acc)])
    print("Test Time ", testTime[np.argmax(acc)])
    print("Accuracy: ", np.max(acc))

'''
OUTPUT DEMO

RANDOM FOREST CLASSIFIER
> Num of estimators:  50
Elapsed time training 28.825665 secs
Elapsed time testing 0.179342 secs
Ensemble accuracy: 0.967000

> Num of estimators:  100
Elapsed time training 56.486980 secs
Elapsed time testing 0.354030 secs
Ensemble accuracy: 0.969429

> Num of estimators:  150
Elapsed time training 85.746600 secs
Elapsed time testing 0.523751 secs
Ensemble accuracy: 0.971286

> Num of estimators:  200
Elapsed time training 114.060088 secs
Elapsed time testing 0.692670 secs
Ensemble accuracy: 0.971429

> Num of estimators:  250
Elapsed time training 142.863594 secs
Elapsed time testing 0.876005 secs
Ensemble accuracy: 0.970571

> Num of estimators:  300
Elapsed time training 182.422549 secs
Elapsed time testing 1.100109 secs
Ensemble accuracy: 0.970429


**-- Best ACC stats --**
Num of estimators:  200
Train Time:  114.06008791923523
Test Time  0.6926703453063965
Accuracy:  0.9714285714285714

'''