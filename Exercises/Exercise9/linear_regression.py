import numpy as np
import time
import matplotlib.pyplot as plt

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
    np.random.seed(2020)
    plt.close('all')
    print('Solar particle dataset')
    X = np.load('particles_X.npy')
    y = np.load('particles_y.npy')
    
    X=np.hstack((X,X**3))
    
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    '''
    #1 - without np.hstacl
    #Divide each column by its std    
    s = np.std(X_train,axis=0)
    X_train = X_train/s
    X_test = X_test/s
    '''
    
    
    model = linear_regression()
    
    start = time.time()
    model.fit(X_train, y_train)
    print(model.W)
    elapsed_time = time.time()-start
    
    
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean squared error: {0:.6f} '.format(mse(pred,y_test)))
        
    plt.plot(y_test,pred,'.')
    
    