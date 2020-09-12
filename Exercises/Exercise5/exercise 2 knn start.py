import numpy as np
import time
import math

class knn(object):
    def __init__(self,k=3,weighted=True,classify=True, distance = 'Euclidean'):  
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
    
    def manhattan_distance(self,X_test):
        # Returns 2D array dist
        # where dist[i,j] is the Manhattan distance from training example i to test example j
        dist = np.zeros((self.X_train.shape[0],X_test.shape[0]))
        for i,x in enumerate(X_test):
            dist[:,i] = np.sum(np.abs(self.X_train - x),axis=1)
        return  dist
    
    def fit(self,x,y): 
        # K-nn just stores the training data
        self.X_train = x
        self.y_train = y
    
    def predict(self,x_test):
        if self.distance=='Euclidean':
            dist  = self.euclidean_distance(X_test)     
            dist = dist**2
        else:
            dist  = self.manhattan_distance(X_test)     
        
        nn = np.argsort(dist,axis=0)[:self.k]
        dist = np.sort(dist,axis=0)[:self.k]
        ind = np.arange(len(y_test))
        if self.weighted:
            w = 1/(dist+1e-10)
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
    
def root_kd(X):
    att = np.argmax(np.std(X,axis=0))
    return np.argsort(X[:,att])[(X.shape[0]+1)//2],att
 
def nn_graph(X,k):
    dist = np.sum(X**2,axis=1).reshape(-1,1) - 2*np.matmul(X,X.T) + np.sum(X.T**2,axis=0).reshape(1,-1) 
    nn = np.argsort(dist,axis=1)
    return nn[:,1:k+1]   
 
def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]    

'''
My implementation of nearest_neighbors(X,x,k) returns 
the indices of the k-nearest neighbors of example x in dataset X.

    1. Let x be the test sample
    2. Find euc dist between test sample  x and all train samples 
    3. Get k closest distances of x from X

'''
def nearest_neighbors(X,x,k):
    
    #Step 1 is already done since it is passed as parameter 
     
    #will contain len(X) euclidean distances - will get first 5 after sorting all of them later
    allDistances = np.zeros(len(X))
    
    #Step 2
    for i in range(len(X)):
        
        dist = math.sqrt(np.sum(pow(abs(x - X[i]),2)))
        allDistances[i] = dist

    #Step 3 - take out the first sample because it is 0 which is the sample comparing it to itself when sorted and get following k values
    nn = np.argsort(allDistances)[1:k+1]
    
    #My nn graph is done ; Is an np array with the k smallest euclidean distances (k closest neighbors) to x
    return nn
    
def graph_nearest_neighbors(x,G,k,r=20,t=20):
    # Returns the indices of the k-nearest neighbors of x
    return np.zeros(k,dtype=int)

if __name__ == "__main__":  
    
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load('mnist_y.npy')
    
    n = 10000      # Use a few examples - The bigger the n the more accurate your readings will be
    
    ind = np.random.permutation(len(y))
    X=X[ind[:n]]
    y=y[ind[:n]]
    
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    '''
    print('k-nn using Euclidean distance')
    model = knn(distance = 'Euclidean')
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    print('k-nn using Manhattan distance')
    model = knn(distance = 'Manhattan')
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
    
    r, att = root_kd(X_train)
    print('kd-tree root example: {}, attribute: {}, threshold: {}'.format(x,att,X_train[r,att]))
    ''' 

    
    nng =  nn_graph(X_train,10)
    
    sample = np.random.randint(X_train.shape[0], size=10)
    
    for n in sample:
        print('Example',n, 'class',y_train[n])
        print('Neighbors class')
        for a in y_train[nng[n]]:
            print(a,end=' ')
        print()
        
    #gets 10 random samples(examples) from the n sized test set 
    sample = np.random.randint(X_test.shape[0], size=10)
    k=5
    
    print("************TASK 1 FROM PDF************")
    print('Nearest neighbors using exhaustive search')
    for i in sample:
        nn = nearest_neighbors(X_train,X_test[i],k)
        
        #Modified to give sample values(0-9) instead of indices       
        
        #print('Nearest neighbors of test example',i)#Fuentes
        print('Nearest neighbors of test example', y_test[i])#Modified
        for a in nn:
            #print(a,end=' ')#Fuents
            print(y_train[a],end=' ')#Modified
        print()
    print('**********************************')
    
    '''
    print('Nearest neighbors using graph approximation')    
    for i in sample:
        nn = graph_nearest_neighbors(X_test[i],nng,k) 
        print('Nearest neighbors of test example',i)
        for a in nn:
            print(a,end=' ')
        print()
    ''' 
    
    
    
    