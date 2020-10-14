import numpy as np
from utils import *
import math 

def sigma(X,W):
    z = np.matmul(X,W.T)
    return (1/(1+np.exp(-z)))

def logistic_regression(X,y,batch_size=512,lr=0.01, tol =0.001,max_it = 1000,display_period=10):
    batches_per_epoch = X.shape[0]//batch_size
    acc_list, mse_list = [], []
    n, m = X.shape                  # the training set has n examples and m attributes
    y_oh = one_hot(y)
    W =  (np.random.random((y_oh.shape[1],m))-0.5)/100    # w is kxm
    for i in range(1,max_it+1):
        ind = np.random.permutation(X.shape[0])%batches_per_epoch
        for b in range(batches_per_epoch):
            batch = (ind == b)
            S = sigma(X[batch],W)          
            Error = (S - y_oh[batch])         
            G = (Error*S*(1-S)).T
            Gradient = np.matmul(G,X[batch])/batch_size   
            W = W - lr*Gradient
        S = sigma(X,W)          
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
    display_probabilities(W[:,:-1])
    return W

plt.close('all')

data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exams\\Exam1\\'  # Use your own path here

X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
y = np.load(data_path+'mnist_y.npy')

X = np.hstack((X,np.ones((X.shape[0],1))))

X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)

W = logistic_regression(X_train,y_train,batch_size= 512,lr=1, tol =0.001,display_period=1,max_it = 20)

S_test = sigma(X_test,W)

pred_test =  np.argmax(S_test,axis=1)

cm = confusion_matrix(y_test,pred_test)   
    
print('Test MSE {:.6f}, Test accuracy: {:.6f}'.format(mse(S_test, one_hot(y_test)),accuracy(pred_test,y_test)))
print(cm)
 