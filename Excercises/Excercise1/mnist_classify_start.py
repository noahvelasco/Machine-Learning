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

#>>>>>>>>>Modified split_train_test
def split_train_test(X,y,percent_train=0.9):
    """
    Steps to splitting between training and testing
    1. Randomize all the samples in the sample space = 'ind' assignment below
    2. Choose the amount of samples you want to train from that randomization = train_indices array using percent_train parameter
    3. Get the left over samples from the randomization to use for testing =
    """
    """
    #USED FOR TESTING
    print('>>Train_Indices = a nummpy array containing 90% of values from (randomly):\n ', train_indices)
    print('>>Length of Train Indices ', percent_train*100 , '% indices:\n ',len(train_indices))

    print('--Test_Indices = a numpy array that contains all the left over indices that train_indices did not use: \n', test_indices)
    print('--Length of Test Indices = remaing ', (1-percent_train)*100  ,'% of indices: \n', len(test_indices))
    """
    #create a random arrangement of numbers from 0 to sample size ; each value represents an index
    indices = np.random.permutation(X.shape[0])

    #get amount of samples you to train (percent train) from all the indices
    train_indices = indices[0:int(len(indices)*percent_train)]

    #get remaining 10% for testing
    test_indices = indices[int(len(indices)*percent_train):]

    train = train_indices # 90% of the random indices from - each index is an image
    test = test_indices   # Remaing 10% of the total indices

    print()

    return X[train],X[test], y[train], y[test]

'''
    find_means() - used for alg 1
    Goal - find the mean image in each of the classes m0,m1,...m9
            Return an array of 10 slots where each slot represents the samples mean image

    Steps:
        1. - Create a 'means' 3D array to hold the mean of all classes (6300,28,28) - Is ordered -> index zero = mean of zero samples
        2. - Calculate how many there are of each class
        3. - Find the sum of each samples respective 28x28 slot and save to mean
        4. - divide each slot by the total class
'''
def find_means(X,y):
    

    '''
    X_train (X) are all 90% of random samples (not sample indices)
    y_train (y) are the corresponding outputs of the 90% samples
    '''
    #Step 1
    means = []
    for i in range(10):
        m = np.zeros((28,28),dtype=int)
        means.append(m)

    #Step 2
    total_classes = np.zeros(10 , dtype=int)

    for i in y:
        total_classes[i] += 1

    #print("Total classes: ", total_classes, " | Total sum: " , sum(total_classes))

    #Go through each sample - using len so that i can access sample in 'y' aswell
    for i in range(len(X)):

        #go through each value and add it to the mean array in numerical order
        for j in range(28):

            for k in range(28):

                #y[i] is the output for sample X[i] ; means[y[i]] indicates putting the sums in order in means array
                means[y[i]][j][k] += X[i][j][k]

    for i in range(10):
        means[i] = means[i] / total_classes[i]

    return means

#accuracy() - calcs the accuracy given the predicted and expected
def accuracy(y_true,y_pred):

    #y_true and y_pred are same size and contain values indicating classifications
    similar = 0

    for i in range(len(y_true)):

        if y_true[i] == y_pred[i]:
            similar +=1

    return similar/len(y_true)

'''
    For every image xi in the test set, find its Euclidean distance to each of the means,
    and assign it to the class of the closest mean.

    Steps:
        1. Turn 'means' into a numpy array
        2. Create a new array that will the store the indices of the closest match (pred)
        3. Find euclidean distance of each one test image and means - the one closest to 0 is closest match - similar to find min in list
        4. Use pred to store index of closest match
        5. Return pred

'''

def classify_nearest_mean(means, X_test):

    m = np.array(means)
    pred =  np.zeros(len(X_test))#contains len(X_test) slots to store all predictions

    for i in range(len(X_test)):

        #assign closest match (first index from 'm') - indices because will help us store for accuracy later
        match = 0
        for j in range(len(m)):

            #find euclidean distance of each sample from X to each sample from means and store closest match(means index) into pred
            dist = np.linalg.norm(X_test[i] - m[j])

            if dist < np.linalg.norm(X_test[i] - m[match]):
                match = j

        #Save the predicted value in the corresponding index as X_test
        pred[i] = match

        #To test this look at variable explorer
        #print("Sample ",i, " from X_test was closest to mean sample classification value: ", match)

    return np.array(pred)

'''
    classify_nearest_example() - For every image xi in the test set, find its Euclidean distance to each
                                 image in the training set and assign it to the class of the closest image.

    Steps:
        1. Create array 'pred' that will contain indices, where each indice represents a number classification
        2. For every image in test set, find its euclidean distance to each image in the training set
            and insert index of closest image into pred
'''

def classify_nearest_example(X_train,y_train, X_test):

    pred =  np.zeros(len(X_test),dtype=int)

    #Get sample from test set
    for i in range(len(X_test)):

        #set first index of train set to be the match TEMPORARLY - will update when starts comparing
        match = 0

        #Get sample from training set
        for j in range(len(X_train)):

            #find euclidean distance of each sample from test to each sample from train and store closest match(y_train value at index=match) into pred
            dist = np.linalg.norm(X_test[i] - X_train[j])

            if dist < np.linalg.norm(X_test[i] - X_train[match]):
                match = j

            #Save the predicted value in the corresponding index as X_test
            pred[i] = y_train[match]

        #To test this look at variable explorer
        #print("Sample ",i, " from X_test was closest to X_train sample classification: ", y_train[match])

    return np.array(pred)

if __name__ == '__main__':
    # Read the data
    X = np.load('mnist_X.npy')
    y = np.load('mnist_y.npy')

    # Plot some of the data - Comment section out when developing your program
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

    print('Evaluating Algorithm 2')
    start = time.time()
    p2 = classify_nearest_example(X_train,y_train, X_test)
    elapsed_time = time.time() - start
    print('Accuracy:',accuracy(y_test,p2))
    print('Elapsed time: ',elapsed_time)

'''
Fuentes' program output:
Evaluating Algorithm 1
Accuracy: 0.8172857142857143
Elapsed time:  0.38397216796875
Evaluating Algorithm 2
Accuracy: 0.9747142857142858
Elapsed time:  1676.9185092449188
'''
