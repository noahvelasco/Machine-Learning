import numpy as np
import time
import matplotlib.pyplot as plt

'''
Goals:

1. Using the particles dataset, determine experimentally if linear regression is sensitive to attribute scaling by comparing results obtain with the original dataset and with the attributes scaled by the standard deviation.
2. Expand the particles dataset by using the squares of the original features as additional features and evaluate the performance of linear regression (you should have 10 features in this case).
3. Expand the particles dataset by using the products of the original features as additional features and evaluate the performance of linear regression (you should have 20 features in this case).
'''

class linear_regression(object):
    def __init__(self):  
        self.W = None
    
    def fit(self,X,y): 
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        self.W = np.matmul(np.linalg.pinv(X1),y.reshape(y.shape[0],-1))
        
    def predict(self,X):
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        p = np.matmul(X1,self.W)
        if p.shape[1]==1:
            p = p.reshape(-1)
        return p
        
def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]    
    
def mse(p,y):
    return np.mean((p-y)**2)
    
def onehot(y):
    oh = np.zeros((len(y),np.amax(y)+1)) 
    oh[np.arange(len(y)),y]=1
    return oh             
    
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

if __name__ == "__main__":  
   
    plt.close('all')
    print('Solar particle dataset')
    X = np.load('particles_X.npy')
    y = np.load('particles_y.npy')
    
    print("\n-- Without scaling --")
    
    np.random.seed(2020)
    #print(X.shape) #(1774943, 5)
    X_train, X_test, y_train, y_test = split_train_test(X,y)   
    
    model = linear_regression()
    
    start = time.time()
    model.fit(X_train, y_train)
    print("> W shape: " , model.W.shape)#vector of  6,1 where each value 
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean squared error: {0:.6f} '.format(mse(pred,y_test)))
    
    #------------------ 1. Scale using Standard Deviation ---------------------
    print("\n-- Scaling using standard deviation --")
    np.random.seed(2020)
    X_train, X_test, y_train, y_test = split_train_test(X,y)   
    
    #scale then fit
    s = np.std(X_train,axis=0)#s has size 5
    X_train = X_train/s
    X_test = X_test/s
    
    model = linear_regression()
    
    start = time.time()
    model.fit(X_train, y_train)
    print("> W shape: " , model.W.shape)#vector of  6,1 where each value 
    elapsed_time = time.time()-start
    
    
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean squared error: {0:.6f} '.format(mse(pred,y_test)))
        
    #plt.plot(y_test,pred,'.')
    
    #------------------ 2. Scale using squares of original features------------
    print("\n-- Scaling using the square of original features --")
    np.random.seed(2020)
    X_new = np.hstack((X,X**2))
    #print(X_new) #(1774943, 10)
    X_train, X_test, y_train, y_test = split_train_test(X_new,y)
    
    model = linear_regression()
    
    start = time.time()
    model.fit(X_train, y_train)
    print( "> W shape: " , model.W.shape)#vector of  10,1 where each value 
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean squared error: {0:.6f} '.format(mse(pred,y_test)))
    
    #plt.plot(y_test,pred,'*')
    
    #-------------------3. Scale to use products of original features ---------   
     
    print("\n-- Scaling using the products of original features --")
    np.random.seed(2020)
    #modify X to include products so that you can stack it to X
    cols = X.shape[1]
    for i in range(cols):
        X = np.hstack((X,X[:,i:cols]*X[:,i].reshape(-1,1)))
        
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    model = linear_regression()
    
    start = time.time()
    model.fit(X_train, y_train)
    print( "> W shape: " , model.W.shape)#vector of  15,1 where each value 
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean squared error: {0:.6f} '.format(mse(pred,y_test)))
    
    '''
Solar particle dataset

-- Without scaling --
> W shape:  (6, 1)
Elapsed_time training  0.337837 
Elapsed_time testing  0.008996 
Mean squared error: 0.043542 

-- Scaling using standard deviation --
> W shape:  (6, 1)
Elapsed_time training  0.306853 
Elapsed_time testing  0.006999 
Mean squared error: 0.043542 

-- Scaling using the square of original features --
> W shape:  (11, 1)
Elapsed_time training  0.714658 
Elapsed_time testing  0.011994 
Mean squared error: 0.039720 

-- Scaling using the products of original features --
> W shape:  (21, 1)
Elapsed_time training  1.666297 
Elapsed_time testing  0.018991 
Mean squared error: 0.039292
    '''