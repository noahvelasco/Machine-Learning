import numpy as np
import time
import math

'''
Goal of this code was to modify starter code by adding:
    1 - Manhattan distance function
'''
class knn(object):
    def __init__(self,k=3,weighted=True,classify=True, distance='Manhattan'):  
        self.k = k
        self.weighted = weighted
        self.classify = classify
        self.distance = distance
    
    def euclidean_distance(self,X_test):
        # Returns 2D array dist
        # where dist[i,j] is the Euclidean distance from training example i to test example j
        dist = np.sum(self.X_train**2,axis=1).reshape(-1,1) # dist = X_train**2
        dist = dist - 2*np.matmul(self.X_train,X_test.T) # dist = X_train**2  - 2*X_train*X_test
        dist = dist + np.sum(X_test.T**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
        dist = np.sqrt(dist) 
        return  dist
    
    '''
    manhattan_distance() - Goal of this function is to calculate the Manhattan distance between
                            each train sample to each test sample and log those results into a 
                            2D matrix called 'dist'. dist[i,j] is the Manhattan distance from 
                            training example i to test example j 
    
                            Returns a 2D array
    
    '''
    def manhattan_distance(self,X_test):

        #Create dist[i,j] where each row reps a unique sample from train, and each column each reps a test sampe. 
        dist = np.zeros((self.X_train.shape[0], X_test.shape[0]))
        
        #print(dist.shape)# (9000 , 1000) if n=10,000
        for i in range(len(self.X_train)):
            for j in range(len(X_test)):
                               
                '''
                #Both ways to calculate euclideann distance | euc1 = euc2 |  - 'euc2' is used to derive manhattan equation
                euc1 = np.linalg.norm(self.X_train[i] - X_test[j])
                euc2 = math.sqrt(np.sum(pow( abs(self.X_train[i] - X_test[j]) , 2 )   ))
                '''  
                
                #Calculate manhattan distance
                dist[i][j] =  np.sum(abs(self.X_train[i] - X_test[j]))
    
        return  dist
    
    def fit(self,x,y): 
        # K-nn just stores the training data
        self.X_train = x
        self.y_train = y
    
    def predict(self,x_test):
        if self.distance=='Euclidean':
            dist  = self.euclidean_distance(x_test)     
        if self.distance == 'Manhattan':
            dist  = self.manhattan_distance(x_test)     
        
        nn = np.argsort(dist,axis=0)[:self.k]
        dist = np.sort(dist,axis=0)[:self.k]
        ind = np.arange(len(y_test))
        if self.weighted:
            w = 1/(dist**2+1e-10)
            sum_w = np.sum(w,axis=0)
            w = w/sum_w
        else:
            w = np.zeros_like(nn,dtype=np.float32)+1/self.k
        if self.classify:
            vote = np.zeros((len(y_test),np.max(self.y_train)+1))
            for i in range(self.k):
                vote[ind,self.y_train[nn[i,:]]] += w[i,:]
            pred = np.argmax(vote,axis=1)
        else:
            pred = np.sum(self.y_train[nn]*w,axis=0)
        return pred

def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]    
    
if __name__ == "__main__":  
    print('MNIST dataset')
   
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load('mnist_y.npy')
    
    n = X.shape[0] # Use all examples
    n = 10000      # Use a few examples
    
    ind = np.random.permutation(len(y))
    X=X[ind[:n]]
    y=y[ind[:n]]
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    #print(">>> Xtrain length - ", len(X_train)) #Is 9000 if n = 10,000
    #print(">>> ytrain length - ", len(y_train)) #Is 9000 if n = 10,000
    #print(">>> X_test length - ", len(X_test)) #Is 1,000 if n = 10,000

    #---------- MODEL 1  = EUCLIDEAN? ---------------
    model = knn()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    #---------- MODEL 2 = MANHATTAN --------------
    model = knn(weighted=False)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    
    
    
'''
>>> AFTER FINISHING ONLY MANHATTAN DISTANCE
    
MNIST dataset
Elapsed_time training  0.000000 
Elapsed_time testing  131.526172 
Accuracy: 0.934
Elapsed_time training  0.000000 
Elapsed_time testing  150.901918 
Accuracy: 0.931

'''
    
    