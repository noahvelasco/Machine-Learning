import numpy as np
import time
import knn
import linear_regression as lr
import naive_bayes as nb


def most_important_feature(X,y):
    
    avg_features = np.sum(X,axis=0)/X.shape[0]
    
    

def most_common_error(y,p):

    #Create confusion matrix    
    conf_matrix = confusion_matrix(y,p)
        
    #print(conf_matrix)#for testing
   
    #temp store indices of confused value - avoid diagonal since they have large values
    maxActInd=0
    maxPredInd=1
    #find the most confused classification and store the indices
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i != j  and (conf_matrix[i][j] > conf_matrix[maxActInd][maxPredInd]):
                maxActInd=i
                maxPredInd=j
    
    return maxActInd,maxPredInd

def remove_constant_attributes(X):
    #Cont Attribute?
    
    
    
    return X

def confusion_matrix(y_true,y_pred):
    cm = np.zeros((np.amax(y_true)+1,np.amax(y_true)+1),dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i],y_pred[i]]+=1
    return cm
   
def split_train_test(X,y,percent_train=0.9,seed=None):
    if seed!=None:
        np.random.seed(seed)
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]    
    
if __name__ == "__main__":  
   
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load('mnist_y.npy')
    X = X[::10]
    y = y[::10]
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=2020)
    
    model = knn.knn()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    act,pre = most_common_error(y_test,pred)
    print('Most common error: actual digit = {}, predicted digit = {}'.format(act,pre))
    
    #TODO
    X = remove_constant_attributes(X)
    print('Shape of data array after removing constant attributes:',X.shape)
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=2020)
    
    model = knn.knn()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    X = np.load('particles_X.npy')
    y = np.load('particles_y.npy')
    
    #TODO
    imp = most_important_feature(X,y)
    print('Most important feature:', imp)   
    
    