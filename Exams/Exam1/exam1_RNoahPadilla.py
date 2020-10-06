import numpy as np
import time
import knn
import linear_regression as lr
import naive_bayes_real_attributes as nb
from utils import *

''' 
@author: R Noah Padilla 
'''

def most_important_feature_2_classes(X_train, X_test, y_train, y_test, c1,c2,n_feat):
    # Return the list with the n_feat feature for which the mean value of elments of class c1 is most different from the mean value of elments of class c2
    # See figre for exampls 
    model = nb.naive_bayes()
    model.fit(X_train, y_train)
    display_probabilities(model.mean_att_given_class) # Display mean atribute values for all classes
    
    '''
    Compare the means of each attribute - two means > each for c1 and c2 so m1=mean1 and m2=mean2
    
    The higher the diff in means for each attribute the more different they are for that attribute
    
    get means difference of each attribute and store them into a list, get the top n_feat indices
    [0,4,5,255,6,34]
    '''
    att_mean_diffs = np.zeros(X_train.shape[1])

    #print(model.mean_att_given_class.shape) #for mnist = (10,784)
   
    for i in range(model.mean_att_given_class.shape[1]):
        att_mean_diffs[i] = (model.mean_att_given_class[c1][i] - model.mean_att_given_class[c2][i])**2
    
    n_feats_indices = np.zeros(n_feat)
    
    for j in range(n_feat):
        #Get the top n_feats - the max mean is an n_feat
        maxIndex = 0
        for i in range(len(att_mean_diffs)):
            if (att_mean_diffs[i] > att_mean_diffs[maxIndex]) and (i not in n_feats_indices): 
                maxIndex = i
        
        n_feats_indices[j] =  maxIndex  
    
    return n_feats_indices

#skipped
def most_important_feature_leave_one_out(X_train, X_test, y_train, y_test):
    '''
    Use linear regression to fi
nd the best feature to predict the target function. For a dataset with n features,
    we train and evaluate n models. Model 0 leaves feature 0 out for training and testing, model 1 leaves feature 1 out,
    and so on. Then we return the index of the feature that was left out in the worst-performing model.
    '''
    mse_list = np.zeros(X_train.shape[1])
    model = lr.linear_regression()
    
    for i in range(len(mse_list)):
        print()
    
        #Do for hw
    return np.argmax(mse_list)
    
def find_best_k(X_train, X_test, y_train, y_test):
    '''
    Find best k by incrementing model.k in a for loop and predict and store values and lastly find argmax
    '''
    model = knn.knn(distance = 'Euclidean')
    model.fit(X_train, y_train)
    
    #NOTE: Indexing will be offset by 1
    k_vals = np.zeros(15)#index 0 = accuracy when k =1 | index 1 = accuracy when k = 2
    
    for i in range(len(k_vals)):
        model.k = i+1 #start at 1 
        pred = model.predict(X_test)
        k_vals[i] = np.sum(pred==y_test)/len(y_test)
    #print(k_vals)
    
    #print out all the values along with k like Fuentes output
    for i in range(len(k_vals)): 
        print('Accuracy:{0:.6f} '.format(k_vals[i]), "when k = ", i+1)
    
    return np.argmax(k_vals)+1 #we add 1 because the indexing if offset, ex. index 0 implies k=1 so if argmax=0 then it means k =argmax+1

if __name__ == "__main__":  
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exams\\Exam1\\'  # Use your own path here
    #data_path =''                        # Use this if data is in the same directory as program
    plt.close('all')
      
    print('\n############# Question 1 ###################')
    X = np.load(data_path+'mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load(data_path+'mnist_y.npy')
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    c1 = 4
    c2 = 9
    n_feat = 3
    imp = most_important_feature_2_classes(X_train, X_test, y_train, y_test, c1,c2,n_feat) 
    print('Most important features to distinguish clases {} and {} are: {}'.format(c1,c2,imp))  
    
    #-------------------------------------------------------------------------------------------------------------SKIP 2
    print('\n############# Question 2 ###################')
    X = np.load(data_path+'particles_X.npy').astype(np.float32) 
    y = np.load(data_path+'particles_y.npy').astype(np.float32) 
    X = np.hstack((X,X*X))
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    imp = most_important_feature_leave_one_out(X_train, X_test, y_train, y_test)
    print('Most important feature is', imp)   
    
    print('\n############# Question 3 ###################')
    n=100
    X = np.load(data_path+'mnist_X.npy').astype(np.float32).reshape(-1,28*28)[::n] # Just pick a small subset of the dataset
    y = np.load(data_path+'mnist_y.npy')[::n]
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    k = find_best_k(X_train, X_test, y_train, y_test)
    print('Best value of k is {}'.format(k))
    
    
'''
Program output:
    
############# Question 1 ###################
Most important features to distinguish clases 4 and 9 are [211 210 212]

############# Question 2 ###################
Error using all attributes except 0 = 0.039772841053048055
Error using all attributes except 1 = 0.03975812552281049
Error using all attributes except 2 = 0.039770929244002204
Error using all attributes except 3 = 0.0399599926802723
Error using all attributes except 4 = 0.04167097321171547
Error using all attributes except 5 = 0.03971670909229692
Error using all attributes except 6 = 0.039700522541145124
Error using all attributes except 7 = 0.039687981921367854
Error using all attributes except 8 = 0.039887622356008526
Error using all attributes except 9 = 0.04059448635403209
Most important feature: 4

############# Question 3 ###################
Accuracy: 0.857143 with k=1
Accuracy: 0.857143 with k=2
Accuracy: 0.871429 with k=3
Accuracy: 0.857143 with k=4
Accuracy: 0.857143 with k=5
Accuracy: 0.842857 with k=6
Accuracy: 0.857143 with k=7
Accuracy: 0.800000 with k=8
Accuracy: 0.828571 with k=9
Accuracy: 0.814286 with k=10
Accuracy: 0.800000 with k=11
Accuracy: 0.785714 with k=12
Accuracy: 0.771429 with k=13
Accuracy: 0.771429 with k=14
Accuracy: 0.785714 with k=15
Best value of k is 3   
'''