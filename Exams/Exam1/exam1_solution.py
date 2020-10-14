import numpy as np
import time
import knn
import linear_regression as lr
import naive_bayes_real_attributes as nb
from utils import *

class knn_q4(object):
    def __init__(self,k=3):  
        self.k = k
        
    def euclidean_distance(self,X_test):
        # Returns 2D array dist
        # where dist[i,j] is the Euclidean distance from training example i to test example j
        dist = np.sum(self.X_train**2,axis=1).reshape(-1,1) # dist = X_train**2
        dist = dist - 2*np.matmul(self.X_train,X_test.T) # dist = X_train**2  - 2*X_train*X_test
        dist = dist + np.sum(X_test.T**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
        dist = np.sqrt(dist) 
        return  dist
    
    def fit(self,x,y): 
        # K-nn just stores the training data
        self.X_train = x
        self.y_train = y
        self.num_classes = np.max(self.y_train)+1
    
    def predict(self,X_test):
        dist  = self.euclidean_distance(X_test)     
        dist = dist**2
        nn = np.argsort(dist,axis=0)[:self.k]
        dist = np.sort(dist,axis=0)[:self.k]
        ind = np.arange(len(X_test))
        w = 1/(dist+1e-10)
        sum_w = np.sum(w,axis=0)
        w = w/sum_w
        vote = np.zeros((len(X_test),self.num_classes))
        for i in range(self.k):
            vote[ind,self.y_train[nn[i,:]]] += w[i,:]
        pred = np.argmax(vote,axis=1)
        conf = vote[np.arange(len(pred)),pred]
        return pred, conf, vote






def most_important_feature_2_classes(X_train, X_test, y_train, y_test, c1,c2,n_feat):
    # Return the list with the n_feat feature for which the mean value of elments of class c1 is most different from the mean value of elments of class c2
    # See figre for exampls 
    model = nb.naive_bayes()
    model.fit(X_train, y_train)
    display_probabilities(model.mean_att_given_class) # Display mean atribute values for all classes
    m = model.mean_att_given_class
    diff = np.abs(m[c1]-m[c2])
    k = np.argsort(-diff)
    return k[:n_feat]

def most_important_feature_leave_one_out(X_train, X_test, y_train, y_test):
    mse_list = []
    model = lr.linear_regression()
    for i in range(X_train.shape[1]):
        L = list(np.arange(X.shape[1]))
        L.pop(i)
        model.fit(X_train[:,L], y_train)
        pred = model.predict(X_test[:,L])
        m = mse(pred,y_test)
        print('Error using all attributes except',i,'=',m)
        mse_list.append(m)
    return np.argmax(np.array(mse_list))

def find_best_k(X_train, X_test, y_train, y_test):
    max_k = 35
    ks = np.zeros(max_k+1)
    for k in range(1,max_k):
        model = knn.knn(k=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        a = accuracy(pred,y_test)
        print('Accuracy: {:.6f} with k={}'.format(accuracy(pred,y_test),k))
        ks[k]=a
    return np.argmax(np.array(ks))

def pred_and_confidence(X_train, X_test, y_train, y_test):
    max_k = 35
    model = knn_q4()
    model.fit(X_train, y_train)
    pred, conf, v = model.predict(X_test)
    return pred, conf, v



if __name__ == "__main__":  
    data_path = 'C:\\Users\\OFuentes\\Documents\\Research\\data\\'  # Use your own path here
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
    
    print('\n############# Question 4 ###################')
    n=5
    X = np.load(data_path+'mnist_X.npy').astype(np.float32).reshape(-1,28*28)[::n] # Just pick a small subset of the dataset
    y = np.load(data_path+'mnist_y.npy')[::n]
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    pred, conf,v = pred_and_confidence(X_train, X_test, y_train, y_test)
    print('Accuracy: {0:.6f}'.format(accuracy(pred,y_test)))
    print('Mean confidence in all predictions: {0:.6f}'.format(np.mean(conf)))
    print('Mean confidence in correct predictions: {0:.6f}'.format(np.mean(conf[pred==y_test])))
    print('Mean confidence in wrong predictions: {0:.6f}'.format(np.mean(conf[pred!=y_test])))
    
    
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