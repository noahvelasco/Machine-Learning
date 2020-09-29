import numpy as np
import time
import matplotlib.pyplot as plt
import knn

'''
Goals:
    
    1. Compare the performance of linear regression and k-nn to predict solar particle flux.
    2. Adapt the linear regression algorithm to work as a classifier and use it to classify the MNIST dataset.
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

#Used for regression classification - Builds an array with all 0's but add '1' where position of 1 is specified from y
def onehot(y):

    oh = np.zeros((len(y), np.amax(y)+1)) 
    oh[np.arange(len(y)),y]=1

    return oh             
    
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

if __name__ == "__main__":  
    
    plt.close('all')
    n = 10000
    
    print("--- Solar particle dataset using", n,  "samples ---\n")
    
    #Only choose some of the data set since the dataset is so huuuuuuuge
    X = np.load('particles_X.npy')[:n]
    y = np.load('particles_y.npy')[:n]  
    
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    #------------------------------Linear Regression---------------------------
    print("Model using Linear Regression")
    modelLinReg = linear_regression()
    
    start = time.time()
    modelLinReg.fit(X_train, y_train)
    elapsed_time = time.time()-start  
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred_LR = modelLinReg.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean squared error: {0:.6f} '.format(mse(pred_LR,y_test)))
    plt.plot(y_test,pred_LR,'o')
    print()
    
    #----------------------------Linear Regression with KNN--------------------
    print("Model using KNN")
    modelKNN = knn.knn(classify=False, distance = 'Manhattan')
    
    start = time.time()
    modelKNN.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred_KNN = modelKNN.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean squared error: {0:.6f} '.format(mse(pred_KNN,y_test)))
    plt.plot(y_test,pred_KNN,'o')
    print()  
    
    #------------------------- Lin. Reg. to classify MNIST -------------------
    
    print("--- Model using Linear Regression to classify MNIST ---")
    
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load('mnist_y.npy')
    
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    '''
        In order to use Linear regression as a classifier we must follow the steps:
            1. Use 'onehot()' which converts the the y_train data to onehot
            2. Predict the data then get argmax(classification) of each row
    '''
    modelLRC = linear_regression()
    start = time.time()
    oh_y_train = onehot(y_train)    #convert train data to onehot
    modelLRC.fit(X_train,oh_y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()
    #When we predict, modelLRC will return the a nx1 list of classifications - convert predicted to onehot representation of X_test and get argmax(classification)
    pred_LRC = np.argmax(modelLRC.predict(X_test),axis=1)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    #print('Mean squared error: {0:.6f} '.format(mse(pred_LRC,y_test))) #not needed here since we are using a classification approach
    print('Accuracy:',np.sum(pred_LRC==y_test)/len(y_test))
    print()  
 
'''
--- Solar particle dataset using 10000 samples ---

Model using Linear Regression
Elapsed_time training  0.002996 
Elapsed_time testing  0.000000 
Mean squared error: 0.022727 

Model using KNN
Elapsed_time training  0.000000 
Elapsed_time testing  1.828794 
Mean squared error: 0.030389 

--- Model using Linear Regression to classify MNIST ---
Elapsed_time training  8.378688 
Elapsed_time testing  0.029985 
Accuracy: 0.8462857142857143
'''