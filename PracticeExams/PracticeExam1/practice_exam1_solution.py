import numpy as np
import time
import knn
import linear_regression as lr
import naive_bayes_real_attributes as nb
from utils import *

def most_important_feature(X,y):
    s = np.std(X,axis=0)
    X = X/s
    model = lr.linear_regression()
    model.fit(X,y)
    return np.argmax(model.W[:-1])
    return 4

def most_common_error(y,p):
    cm  = confusion_matrix(y,p)
    cm[np.arange(cm.shape[0]),np.arange(cm.shape[0])] = -1
    act, pre = np.unravel_index(np.argmax(cm, axis=None), cm.shape)
    return act, pre

def remove_constant_attributes(X,eps=0):
    return X[:,np.var(X,axis=0)>eps]

def confusion_matrix(y_true,y_pred):
    cm = np.zeros((np.amax(y_true)+1,np.amax(y_true)+1),dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i],y_pred[i]]+=1
    return cm
   
   
    
if __name__ == "__main__":  
    data_path = 'C:\\Users\\OFuentes\\Documents\\Research\\data\\'  # Use your own path here
    #data_path =''
    plt.close('all')
      
    print('\n############# Question 1 ###################')
    X = np.load(data_path+'mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load(data_path+'mnist_y.npy')

    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    print('Linear regression results')
    model = lr.linear_regression()
    model.fit(X_train, onehot(y_train))
    pred = model.predict(X_test)
    pred = np.argmax(pred,axis=1)
    print('Accuracy: {0:.6f}'.format(accuracy(pred,y_test)))
    
    # Verify the dataset with fewer columns yields the same results
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    X = remove_constant_attributes(X)
    print('Datset shape after removing constant attributes:',X.shape)
    print('Linear regression results')
    model = lr.linear_regression()
    model.fit(X_train, onehot(y_train))
    pred = model.predict(X_test)
    pred = np.argmax(pred,axis=1)
    print('Accuracy: {0:.6f}'.format(accuracy(pred,y_test)))
    
    print('\n############# Question 2 ###################')
    act,pre = most_common_error(y_test,pred)
    print('Most common error: actual digit = {}, predicted digit = {}'.format(act,pre))
    
    
    print('\n############# Question 3 ###################')
    X = np.load(data_path+'particles_X.npy').astype(np.float32) # Read only every nth item
    y = np.load(data_path+'particles_y.npy').astype(np.float32) # Read only every nth item
    imp = most_important_feature(X,y)
    print('Most important feature:', imp)   
    
    print('\n############# Question 4 ###################')
    n=100
    X = np.load(data_path+'mnist_X.npy').astype(np.float32).reshape(-1,28*28)[::n] # Just pick a small subset of the dataset
    y = np.load(data_path+'mnist_y.npy')[::n]
    X = remove_constant_attributes(X)
    
    cols = X.shape[1]
    for i in range(cols):
        X = np.hstack((X,X[:,i:cols]*X[:,i].reshape(-1,1)))
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=2020)
    model = nb.naive_bayes()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    