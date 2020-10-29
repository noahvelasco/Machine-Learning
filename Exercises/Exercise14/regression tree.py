import numpy as np
from utils import *
import math 
import time
from sklearn.tree import DecisionTreeRegressor
'''
Author: @R Noah Padilla

Goal - improve the DTree Regressor to improve MSE by modifying the parameters and testing.
        1. Find best max_depth
        2. Find the best  max_leaf_nodes given the best max_depth

'''
if __name__ == "__main__":  
    plt.close('all')
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise14\\'
    
    X = np.load(data_path+'particles_X.npy').astype(np.float32)
    y = np.load(data_path+'particles_y.npy').astype(np.float32)
  
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
      
    '''
    class sklearn.tree.DecisionTreeRegressor(*, criterion='mse', splitter='best', max_depth=None,
                                             min_samples_split=2, min_samples_leaf=1, 
                                             min_weight_fraction_leaf=0.0, max_features=None, 
                                             random_state=None, max_leaf_nodes=None, 
                                             min_impurity_decrease=0.0, min_impurity_split=None, 
                                             presort='deprecated', ccp_alpha=0.0)
    '''
    
    depths = []
    times = []
    mse_train = []
    mse_test = []
    
    for max_depth in range(1,25,1):
        depths.append(max_depth) 
        print('\nMaximum depth:',max_depth)
        model = DecisionTreeRegressor(max_depth=max_depth)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time()-start
        print('Elapsed_time training  {0:.4f} secs'.format(elapsed_time))  
        times.append(elapsed_time)
        
        print('Performance on training set')
        start = time.time()       
        pred = model.predict(X_train)
        elapsed_time = time.time()-start
        print('Elapsed_time testing  {0:.4f} secs'.format(elapsed_time))  
        err = mse(y_train,pred)
        mse_train.append(err)
        print('Mean-squared error: {0:.6f}'.format(err))
              
        print('Performance on test set')
        start = time.time()       
        pred = model.predict(X_test)
        elapsed_time = time.time()-start
        print('Elapsed_time testing  {0:.4f} secs'.format(elapsed_time))   
        err = mse(y_test,pred)
        mse_test.append(err)
        print('Mean-squared error: {0:.6f}'.format(err))
    
    fig, ax = plt.subplots(2,figsize=(6,6))
    
    ax[0].plot(depths,times)
    ax[0].title.set_text('Training time')  
    ax[1].plot(depths,mse_train)
    ax[1].plot(depths,mse_test)  
    ax[1].title.set_text('Mean squared error') 
    
    print("Index of min and value:", np.argmin(mse_test), " | ", np.min(mse_test) )
    #----------------------------------------------------------------------------
    '''
    Get the best depth and use it in this next process where we use max leaf nodes
    
    '''
    best_depth=np.argmin(mse_test)+1 #Get the best depth from the previous run to use it for the next test
    
    depths = []
    max_leafnodes = []
    times = []
    mse_train = []
    mse_test = []
    
    for MaxLeafNodes in range(100,5000,500):
        depths.append(best_depth) 
        max_leafnodes.append(MaxLeafNodes)
        print('\nMaximum depth:',best_depth)
        print('Maximum leaf nodes:', MaxLeafNodes)
        model = DecisionTreeRegressor(max_leaf_nodes=MaxLeafNodes, max_depth=best_depth )#<<<<<<Change this line
        start = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time()-start
        print('Elapsed_time training  {0:.4f} secs'.format(elapsed_time))  
        times.append(elapsed_time)
        
        print('Performance on training set')
        start = time.time()       
        pred = model.predict(X_train)
        elapsed_time = time.time()-start
        print('Elapsed_time testing  {0:.4f} secs'.format(elapsed_time))  
        err = mse(y_train,pred)
        mse_train.append(err)
        print('Mean-squared error: {0:.6f}'.format(err))
              
        print('Performance on test set')
        start = time.time()       
        pred = model.predict(X_test)
        elapsed_time = time.time()-start
        print('Elapsed_time testing  {0:.4f} secs'.format(elapsed_time))   
        err = mse(y_test,pred)
        mse_test.append(err)
        print('Mean-squared error: {0:.6f}'.format(err))
    
    fig, ax = plt.subplots(2,figsize=(6,6))
    
    ax[0].plot(max_leafnodes,times)
    ax[0].title.set_text('Training time')  
    ax[1].plot(max_leafnodes,mse_train)
    ax[1].plot(max_leafnodes,mse_test)  
    ax[1].title.set_text('Mean squared error') 
