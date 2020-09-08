import numpy as np
import matplotlib.pyplot as plt
import time 
import math 

def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

def confusion_matrix(y_true,y_pred):
    cm = np.zeros((np.amax(y_true)+1,np.amax(y_true)+1),dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i],y_pred[i]]+=1
    return cm

def precision(y_true,y_pred,positive_class):
    tp = np.sum((y_true==positive_class) & (y_pred==positive_class))
    pred_p = np.sum(y_pred==positive_class)
    return tp/pred_p

def recall(y_true,y_pred,positive_class):
    tp = np.sum((y_true==positive_class) & (y_pred==positive_class))
    true_p = np.sum(y_true==positive_class)
    return tp/true_p

#classify_nearest_example_fast() - an alg that calculates the closest classification of the test data. Given code. 
def classify_nearest_example_fast(X_train,y_train, X_test):
    # Uses formula (X_train - X_test)**2 = X_train**2 - 2*X_train*X_test + X_test**2
    # Each of the 3 elements can be computed without loops. Third term is not necessary
    # Addition of 3 terms is done with broadcasting without loops
    X_train =  X_train.reshape(-1,784)
    X_test =  X_test.reshape(-1,784).T
    dist = np.sum(X_train**2,axis=1).reshape(-1,1) # dist = X_train**2
    dist = dist - 2*np.matmul(X_train,X_test) # dist = X_train**2  - 2*X_train*X_test
    dist = dist + np.sum(X_test**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
    dist = np.sqrt(dist) #  Not really necessary
    nn = np.argmin(dist,axis=0)
    return y_train[nn]

#Already Done
def classify_nearest_example_loocv(X,y):
    X =  X.reshape(-1,784)
    dist = np.sum(X**2,axis=1).reshape(-1,1) # dist = X_train**2
    dist = dist - 2*np.matmul(X,X.T) # dist = X_train**2  - 2*X_train*X_test
    dist = dist + np.sum(X.T**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
    dist = np.sqrt(dist) #  Not really necessary
    dist[np.arange(dist.shape[0]),np.arange(dist.shape[0])]= math.inf
    nn = np.argmin(dist,axis=0)
    return y[nn]

'''

classify_nearest_example_kfcv() funtion: 

k-fold class validation - A statistical method thats main purpose is to evaluate generalization performance. 
                        Instead of splitting into 75% training and 25% testing we are going to make 
                        'k' groups or folds and train on k-1 groups and test the kth group that was left out. This is an 
                        iterative process so we will test each group at least once and train the remaining 
                        groups. For ex. we will train groups 2,3,4,5 and test group 1 then we will train groups 1,3,4,5
                        and test group 2 and so on. We will repeat the process until we test all the groups.
                        
                        Goal: Use kfcv to predict all the values from x 
                            
                        Given(Parameters passed into this function): 
                            > X = A (n,28,28) array that contains all samples
                            > y = The X array samples' respective outputs
                            > k = 5 indicates the number of groups we are splitting all data into
                            
                        My Algorithm:
                            1. Split X into K-Folds 
                            2. Send Train groups and Test group to given method 'classify_nearest_example_fast()'
                            3. Return pred[] which is an array of predictions from prediction the held out group

'''
def classify_nearest_example_kfcv(X,y,k=5):
    
    #pred will store all the predictions from X
    pred = np.zeros_like(y) 
    
    #A group of groups - each group represents a list numpy array of data
    group_X = [] #np.zeros_like(y)
    
    #A group of groups - Each group represents data outputs in respective order of X groups
    group_y = []#np.zeros_like(y)
    
    #lowerBound monitors what index to start adding values from
    lowerBound = 0
    #upperBound monitors what index to add values up until; counter is
    upperBound =  int(X.shape[0] / (k))
       
    #Step 1
    for i in range(k):
        
        #used for testing - calculates the lower and upper bound indices to choose from
        #print('i = ', i , '| lowB = ', lowerBound , '| upperB = ', upperBound )
       
        group_X.append(X[lowerBound:upperBound])
        group_y.append(y[lowerBound:upperBound])
    
        #update lower and upperbounds
        lowerBound = upperBound
        upperBound +=  int(X.shape[0] / (k))        
    
    '''
    Step 2 - To get groups for training and groups for testing:
        Training: 
            1. Keep track of which group is test group
            2. Get all indices that are not from test group (values range from 0-9,999)
        Testing:
            1. Choose testing group in ascending order
            2. Send that group to classify_nearest_example_fast()
            
        Use index variables to monitor index for training
    
    '''
    #Variable that monitors which group is the test group in group_X; ranges 1-5
    testGroupIndex = 0
    
    #go through all groups 5 times - each time update testGroup
    for i in range(k):
        
        #Variable to monitor which group we are in group_X; ranges from 1-5
        currGroupIndex = 0
       
        #A counter monitoring which index we are at out of 10,000 samples; starts over when choosing a new test group
        indexCounter = 0
        
        #List containing all the indices of all test samples; contains 2,000 values indicating indices
        test = []
        
        #List containing all the indices of all train samples; contains 8,000 values indicating indices
        train = []
        
        #Iterate over all groups and seperate train and test indices
        for j in range(len(group_X)):
            
            #Access each sample in each group ;2000 samples
            for z in range(len(group_X[j])):
                #print(indexCounter)
                if currGroupIndex == testGroupIndex:
                    test.append(indexCounter)
                else:
                    train.append(indexCounter)
                indexCounter += 1
            
            
            currGroupIndex += 1
        
        #Train and test now - this line is executed k times where k=# of groups
        pred[test] = classify_nearest_example_fast(X[train],y[train], X[test])    
        
        #Iterate again and choose the next test group
        testGroupIndex +=1
        
    #Step 3
    return pred

if __name__ == '__main__':
    # Read the data
    np.random.seed(4361) # Set seed to obtain repeatable results
    n = 10000
    X = np.load('mnist_X.npy').astype(np.float32)
    y = np.load('mnist_y.npy')
    ind = np.random.permutation(len(y))
    X=X[ind[:n]]
    y=y[ind[:n]]
    
    
    
    print('Evaluating Algorithm 2')
    start = time.time()
    print('Leave-one-out cross validation')
    p2 = classify_nearest_example_loocv(X,y)
    elapsed_time = time.time() - start 
    print('Accuracy:',accuracy(y,p2))
    print('Elapsed time: ',elapsed_time)
    print('Confusion matrix:')
    print(confusion_matrix(y,p2))
    print('Precision and recall:')
    for i in range(np.amax(y)+1):
        print('Positive class {}, precision: {:.4f}, recall: {:.4f}'.format(i, precision(y,p2,i),recall(y,p2,i)))
    
    #------------------------------------------------------------------------------------------------------------
    print('k-fold cross validation')
    start = time.time()
    p2 = classify_nearest_example_kfcv(X,y)
    elapsed_time = time.time() - start 
    print('Accuracy:',accuracy(y,p2))
    print('Elapsed time: ',elapsed_time)
    print('Confusion matrix:')
    print(confusion_matrix(y,p2))
    print('Precision and recall:')
    for i in range(np.amax(y)+1):
        print('Positive class {}, precision: {:.4f}, recall: {:.4f}'.format(i, precision(y,p2,i),recall(y,p2,i)))

'''
Evaluating Algorithm 2
Leave-one-out cross validation
Accuracy: 0.9522
Elapsed time:  2.4264791011810303
Confusion matrix:
[[ 986    1    0    0    1    4    4    1    2    0]
 [   0 1147    1    1    3    0    1    3    0    3]
 [  11    9  950    4    5    3    4   14    4    6]
 [   0    4    7  981    0   21    2    7   16    3]
 [   0   10    4    0  917    0    3    6    0   41]
 [   5    5    2   27    1  823   14    4    7    5]
 [   2    1    0    0    0    6  955    0    3    0]
 [   1    3    4    1    4    0    0  971    1   23]
 [   8   14    5   19    3   22    5    3  895   10]
 [   4    3    1    4   23    2    1   20    3  897]]
Precision and recall:
Positive class 0, precision: 0.9695, recall: 0.9870
Positive class 1, precision: 0.9582, recall: 0.9896
Positive class 2, precision: 0.9754, recall: 0.9406
Positive class 3, precision: 0.9460, recall: 0.9424
Positive class 4, precision: 0.9582, recall: 0.9348
Positive class 5, precision: 0.9342, recall: 0.9216
Positive class 6, precision: 0.9656, recall: 0.9876
Positive class 7, precision: 0.9436, recall: 0.9633
Positive class 8, precision: 0.9613, recall: 0.9096
Positive class 9, precision: 0.9079, recall: 0.9363
k-fold cross validation
Accuracy: 0.9467
Elapsed time:  4.4121668338775635
Confusion matrix:
[[ 986    1    0    0    1    3    3    1    4    0]
 [   0 1146    1    1    4    0    1    4    0    2]
 [  12   12  938    4    5    4    5   21    3    6]
 [   0    4    8  976    0   22    1    7   16    7]
 [   0   12    3    0  913    0    3    5    0   45]
 [   5    6    2   34    2  811   17    3    7    6]
 [   5    1    0    0    1    6  951    0    3    0]
 [   1    5    4    1    5    0    0  967    1   24]
 [   9   13    4   18    3   25    5    6  888   13]
 [   4    6    2    4   23    1    2   22    3  891]]
Precision and recall:
Positive class 0, precision: 0.9648, recall: 0.9870
Positive class 1, precision: 0.9502, recall: 0.9888
Positive class 2, precision: 0.9751, recall: 0.9287
Positive class 3, precision: 0.9403, recall: 0.9376
Positive class 4, precision: 0.9540, recall: 0.9307
Positive class 5, precision: 0.9300, recall: 0.9082
Positive class 6, precision: 0.9626, recall: 0.9835
Positive class 7, precision: 0.9334, recall: 0.9593
Positive class 8, precision: 0.9600, recall: 0.9024
Positive class 9, precision: 0.8964, recall: 0.9301
''' 