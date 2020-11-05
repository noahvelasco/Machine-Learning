# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:37:08 2020

@author: R Noah Padilla
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
    
    #A place to store the accuracies we are looking for and their respective PC's
    prince_comp = []
    all_acc=[]
    
    #while you still havent gotten .97%
    pc = 1
    while .97 not in all_acc:
        print("PC = ", pc)
        pca = PCA(n_components=pc)
        pca.fit(X_train)
        ev = pca.explained_variance_ratio_
        plt.plot(ev)
        cum_ev = np.cumsum(ev)
        cum_ev = cum_ev/cum_ev[-1]
        plt.plot(cum_ev)
        
        X_train_t = pca.transform(X_train)
        X_test_t = pca.transform(X_test)
        '''
        print('Explained variance:',pca.explained_variance_ratio_)
        print('Data mean:',pca.mean_)
        print('First component :',pca.components_[0]) 
        print('Second component :',pca.components_[1]) 
        '''
        
        '''
        class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, 
                                                    weights='uniform', algorithm='auto', 
                                                    leaf_size=30, p=2, metric='minkowski', 
                                                    metric_params=None, n_jobs=None, **kwargs)
        '''
        
        #model = DecisionTreeClassifier(criterion='entropy',random_state=0)
        model = KNeighborsClassifier(n_neighbors=3)
        
        start = time.time()
        model.fit(X_train_t, y_train)
        elapsed_time_train = time.time()-start
        #print('Elapsed_time training  {0:.6f} secs'.format(elapsed_time_train))  
        start = time.time()
        pred = model.predict(X_test_t)
        elapsed_time_test = time.time()-start
        #print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time_test))
        #print('Accuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
        if (round(accuracy(y_test,pred),2) == .90) or (round(accuracy(y_test,pred),2) == .95) or (round(accuracy(y_test,pred),2) == .97) and ( round(accuracy(y_test,pred),2) not in all_acc) :
            
            print("* PC and ACC: ", pc, " | ", round(accuracy(y_test,pred),2))
            print('* Elapsed_time training  {0:.6f} secs'.format(elapsed_time_train))
            print('* Elapsed_time testing  {0:.6f} secs'.format(elapsed_time_test))
            
            #if we are here than that means we know and will store the PC's for each desired accuracy
            prince_comp.append(pc)
            all_acc.append(round(accuracy(y_test,pred),2))
        
        #update the number of principle components
        pc += 1
        
        
    '''
    >OUTPUT IF MODEL IS DECISION TREE and PC IS 150
    Elapsed_time training  74.612365 secs
    Elapsed_time testing  0.005997 secs
    Accuracy on test set: 0.839857
    Depth:  21
    Leaves:  4199

    
    >OUTPUT IF MODEL IS KNN and PC is 150
    Elapsed_time training  1.392429 secs
    Elapsed_time testing  113.752312 secs
    Accuracy on test set: 0.975286
    
    
    >OUTPUT IF MODEL IS KNN AND PC is 15
    Elapsed_time training  0.149929 secs
    Elapsed_time testing  3.783973 secs
    Accuracy on test set: 0.962286
    
    >OUTPUT IF MODEL IS KNN AND PC IS VARIOUS
    PC =  1
    PC =  2
    PC =  3
    PC =  4
    PC =  5
    PC =  6
    PC =  7
    PC =  8
    * PC and ACC:  8  |  0.9
    * Elapsed_time training  0.071012 secs
    * Elapsed_time testing  0.819000 secs
    PC =  9
    PC =  10
    PC =  11
    PC =  12
    * PC and ACC:  12  |  0.95
    * Elapsed_time training  0.100952 secs
    * Elapsed_time testing  2.187750 secs
    PC =  13
    PC =  14
    PC =  15
    PC =  16
    * PC and ACC:  16  |  0.97
    * Elapsed_time training  0.129938 secs
    * Elapsed_time testing  3.603628 secs
    
    '''