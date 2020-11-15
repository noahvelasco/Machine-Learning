import numpy as np
from utils import *
import math 
import time
from sklearn.tree import DecisionTreeClassifier
 
'''
optimize classifier with paramter modifications
'''
if __name__ == "__main__":  
    plt.close('all')
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise13\\'
    
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)
    
    y = np.load(data_path+'mnist_y.npy')
    thr = 127
    X = (X>thr).astype(int)
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    '''
    class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', 
                                              max_depth=None, min_samples_split=2, 
                                              min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                              max_features=None, random_state=None, max_leaf_nodes=None, 
                                              min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, 
                                              presort='deprecated', ccp_alpha=0.0)
    
    Notes from https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    
    The default values for the parameters controlling the 
    size of the trees (e.g. max_depth, min_samples_leaf, etc.) 
    lead to fully grown and unpruned trees which can potentially be very 
    large on some data sets. To reduce memory consumption, the complexity 
    and size of the trees should be controlled by setting those parameter values.
    
    
    Steps to optimize accuracy and run time
        1. Change model to have criterion with 'entropy' instead of gini
        2. Find model with best depth without min_leaf_samples
        3. Find model with best min_leaf_samples without depth
        4. Find model with best both
    '''
    
    #---------------------------------------- 1 ----------------------------------------
    
    print(">>>>>> Parameters changed: entropy")
    model = DecisionTreeClassifier(criterion='entropy',random_state=0)
    
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} secs'.format(elapsed_time))  
    
    pred = model.predict(X_train)
    print('Accuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
    print('Depth: ', model.get_depth())
    print('Leaves: ', model.get_n_leaves())
    
    # ---------------------------------------- 2 ----------------------------------------
    
    print("--- Parameters changed: entropy, tree depth limit ---")
    DT_depth = 10
    
    while DT_depth <= 50:
        model = DecisionTreeClassifier(criterion='entropy', max_depth=DT_depth , random_state=0)
        
        start = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time()-start
        print('>>> Elapsed_time training  {0:.6f} secs'.format(elapsed_time))  
        
        pred = model.predict(X_train)
        print('Accuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
        
        start = time.time()
        pred = model.predict(X_test)
        elapsed_time = time.time()-start
        print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))   
        print('Accuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
        print('Depth: ', model.get_depth())
        print('Leaves: ', model.get_n_leaves())
        
        DT_depth+=10
    
    # ---------------------------------------- 3 ----------------------------------------
    
    print("--- Parameters changed: entropy, minimum samples per leaf ---")
    minSampLeaves= 1
    
    while minSampLeaves <= 20:
        model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=minSampLeaves, random_state=0)
        
        start = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time()-start
        print('>>> Elapsed_time training  {0:.6f} secs'.format(elapsed_time))  
        
        pred = model.predict(X_train)
        print('Accuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
        
        start = time.time()       
        pred = model.predict(X_test)
        elapsed_time = time.time()-start
        print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))   
        print('Accuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
        print('Depth: ', model.get_depth())
        print('Leaves: ', model.get_n_leaves())
        
        minSampLeaves*=2
    
    #4 - results same as 1 so didnt code it out
    
    #---------------------------------------- 5 ----------------------------------------
    
    print("--- Parameters changed: entropy, max depth and max leaf nodes---")
    mln= 2000
    
    while mln <= 6000:
        print('>>> mln: ',mln)
        model = DecisionTreeClassifier(criterion='entropy', max_depth=20, max_leaf_nodes=mln , random_state=0)
        
        start = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time()-start
        print('Elapsed_time training  {0:.6f} secs'.format(elapsed_time))  
        
        pred = model.predict(X_train)
        print('Accuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
        
        start = time.time()       
        pred = model.predict(X_test)
        elapsed_time = time.time()-start
        print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))   
        print('Accuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
        print('Depth: ', model.get_depth())
        print('Leaves: ', model.get_n_leaves())
        
        mln += 1000
    
'''
-----------Output without changes----------   
Elapsed_time training  17.761801 secs
Accuracy on training set: 1.000000
Elapsed_time testing  0.027988 secs
Accuracy on test set: 0.861429
Depth:  47
Leaves:  5558


-----------Output with 'entropy' instead of gini ----------
Elapsed_time training  11.643139 secs
Accuracy on training set: 1.000000
Elapsed_time testing  0.033979 secs
Accuracy on test set: 0.872429
Depth:  30
Leaves:  5082

'''