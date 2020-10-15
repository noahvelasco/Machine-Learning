import numpy as np
from utils import *
import time

def sigma(X,W):
    z = np.matmul(X,W.T)
    return (1/(1+np.exp(-z)))

class logistic_regression():
    def __init__(self):
        self.W = None
    
    def fit(self,X,y,batch_size=512,lr=0.01, tol =0.001,max_it = 1000,display_period=10):
        
        batches_per_epoch = X.shape[0]//batch_size
        acc_list, mse_list = [], []
        n, m = X.shape                  # the training set has n examples and m attributes
        y_oh = one_hot(y)
        self.W =  (np.random.random((y_oh.shape[1],m))-0.5)/100    # w is kxm
        for i in range(1,max_it+1):
            ind = np.random.permutation(X.shape[0])%batches_per_epoch
            for b in range(batches_per_epoch):
                batch = (ind == b)
                S = sigma(X[batch],self.W)
                Error = (S - y_oh[batch])
                G = (Error*S*(1-S)).T
                Gradient = np.matmul(G,X[batch])/batch_size
                self.W = self.W - lr*Gradient
            S = sigma(X,self.W)
            Error = (S - y_oh)
            pred = np.argmax(S,axis=1)
            acc = accuracy(y,pred)
            acc_list.append(acc)
            mse_train = np.mean(Error**2) 
            mse_list.append(mse_train)
            
            if mse_train<tol:
                break
           
            if i%display_period==0:  
                print('Epoch: {}/{}, mse = {:.6f}, accuracy = {:.4f}'.format(i,max_it,mse_train,acc))
        
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(mse_list)
        ax[0].set_title('Training mean-squared error')
        ax[1].plot(acc_list)
        ax[1].set_title('Training accuracy')
        #display_probabilities(self.W[:,:-1])#used for running this file
        #display_probabilities(self.W[:,:])#used for running mnist_comparison
    
    def predict(self,X):
       S_test = sigma(X,self.W)
       pred_test =  np.argmax(S_test,axis=1)
       return pred_test
        
if __name__ == "__main__":
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exams\\Exam1\\'  # Use your own path here
    
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
    
    X = np.hstack((X,np.ones((X.shape[0],1))))
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    

    print('<<< Logistic Regression Model Modified >>>')
    model = logistic_regression() # Replace previous line by this one
    start = time.time()
    model.fit(X_train,y_train,batch_size= 512,lr=1, tol =0.001,display_period=1,max_it = 20) 
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    cm=confusion_matrix(y_test,pred)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred,y_test)))
    print(cm)

    '''
<<< Logistic Regression Model Modified >>>
Epoch: 1/20, mse = 0.029271, accuracy = 0.8706
Epoch: 2/20, mse = 0.024875, accuracy = 0.8834
Epoch: 3/20, mse = 0.022934, accuracy = 0.8905
Epoch: 4/20, mse = 0.021783, accuracy = 0.8948
Epoch: 5/20, mse = 0.020996, accuracy = 0.8969
Epoch: 6/20, mse = 0.020384, accuracy = 0.8991
Epoch: 7/20, mse = 0.019899, accuracy = 0.9014
Epoch: 8/20, mse = 0.019546, accuracy = 0.9028
Epoch: 9/20, mse = 0.019185, accuracy = 0.9041
Epoch: 10/20, mse = 0.018894, accuracy = 0.9053
Epoch: 11/20, mse = 0.018651, accuracy = 0.9063
Epoch: 12/20, mse = 0.018453, accuracy = 0.9073
Epoch: 13/20, mse = 0.018240, accuracy = 0.9080
Epoch: 14/20, mse = 0.018086, accuracy = 0.9085
Epoch: 15/20, mse = 0.017953, accuracy = 0.9097
Epoch: 16/20, mse = 0.017786, accuracy = 0.9099
Epoch: 17/20, mse = 0.017672, accuracy = 0.9101
Epoch: 18/20, mse = 0.017551, accuracy = 0.9108
Epoch: 19/20, mse = 0.017429, accuracy = 0.9110
Epoch: 20/20, mse = 0.017326, accuracy = 0.9119
Elapsed_time training:  12.145195 secs
Elapsed_time testing: 0.013993 secs
Accuracy: 0.906571
[[649   0   2   1   1   1   3   1  11   1]
 [  1 787   1   4   0   4   1   1  14   1]
 [  3   6 648  12  13   2  12  13  29   4]
 [  4   4  24 639   2  20   3   9  14   9]
 [  1   1   6   1 633   1   6   0   5  19]
 [ 17   8   3  27  17 504  19   5  29   8]
 [ 10   3   3   0   9   5 658   0   4   0]
 [  3   6  12   1   4   0   0 618   3  15]
 [  5  14   8  14   4  14   7   3 599   7]
 [  6   2   4   9  35   6   0  29   5 611]]
    '''

