import numpy as np
import matplotlib.pyplot as plt
import time 

def display_random_digits(X,n=10):
    fig, ax = plt.subplots(1,n,figsize=(n,1))
    for j in range(n):
        ind = np.random.randint(0,X.shape[0])
        ax[j].imshow(X[ind],cmap='gray')
        ax[j].axis('off')
    
def display_means(M):
    fig, ax = plt.subplots(1,10,figsize=(10,1))
    for i in range(10):
        ax[i].imshow(M[i],cmap='gray')
        ax[i].axis('off')    
    
def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]

def find_means(X,y):
    means = []
    for i in range(10):
        m = np.mean(X[y==i],axis=0)
        means.append(m)
    return means

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

def classify_nearest_mean(means, X_test):
    m = np.array(means)
    pred =  []
    for x in X_test:
        d = x.reshape(1,28,28) - means
        p = np.sum(d**2,axis=(1,2))
        pred.append(np.argmin(p))
    return np.array(pred) 

def classify_nearest_example(X_train,y_train, X_test):
    pred =  []
    for x in X_test:
        d = x.reshape(1,28,28) - X_train
        p = np.sum(d**2,axis=(1,2))
        pred.append(y_train[np.argmin(p)])
        if len(pred)==1000:
            break
    return np.array(pred) 

def classify_nearest_example_fast(X_train,y_train, X_test):
    # Uses formula (X_train - X_test)**2 = X_train**2 - 2*X_train*X_test + X_test**2
    # Each of the 3 elements can be computed without loops. Third term is not necessary
    # Addition of 3 terms is done with broadcasting without loops
    X_train =  X_train.reshape(-1,784)
    X_test =  X_test.reshape(-1,784).T
    dist = np.sum(X_train**2,axis=1).reshape(-1,1) # dist = X_train**2
    dist = dist - 2*np.matmul(X_train,X_test) # dist = X_train**2  - 2*X_train*X_test
    dist = dist + np.sum(X_test**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
    nn = np.argmin(dist,axis=0)
    return y_train[nn]


if __name__ == '__main__':
    # Read the data
    X = np.load('mnist_X.npy').astype(np.float32)
    y = np.load('mnist_y.npy')
    
    # Plot some of the data - Comment whole section out when developing your program
    plt.close('all')
    display_random_digits(X,n=10)
    
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    print('Evaluating Algorithm 1')
    start = time.time()
    means = find_means(X_train,y_train)
    display_means(means)
    p1 = classify_nearest_mean(means, X_test)
    elapsed_time = time.time() - start 
    print('Accuracy:',accuracy(y_test,p1))
    print('Elapsed time: ',elapsed_time)
    np.save('y_test_a1',y_test)
    np.save('pred_a1',p1)
    
    print('Evaluating Algorithm 2')
    start = time.time()
    p2 = classify_nearest_example_fast(X_train,y_train, X_test)
    elapsed_time = time.time() - start 
    print('Accuracy:',accuracy(y_test,p2))
    print('Elapsed time: ',elapsed_time)
    np.save('y_test_a2',y_test)
    np.save('pred_a2',p2)
    
   