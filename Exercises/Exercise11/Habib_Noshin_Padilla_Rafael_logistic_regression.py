import numpy as np
from utils import *
import math 

'''
Authors: @R Noah Padilla
         @Noshin R Habib

Goals:
    Given Logistic Regression Machine Learning Model, apply the below 
    improvements to make the model as efficient as possible: 
        
        Adaptive Learning Summary-
            > Reduce learning rate after a fixed number of epochs
            > Reduce learning rate when MSE does not decrease after a certain number of epochs

        Momentum Summary -
            > Keep a running estimate of the gradient, combining the previous estimate and the gradient computed
              from the current 
              
              The idea of momentum-based optimizers is to remember the previous gradients 
              from recent optimization steps and to use them to help to do a better job of 
              choosing the direction to move next, acting less like a drunk student walking 
              downhill and more like a rolling ball. - https://gluon.mxnet.io/chapter06_optimization/momentum-scratch.html
        
        Label Smoothing Summary-
            > y_smooth = y_oh oh(α)+ ൗൗ(1−α)n_classesfor α<1 Notice that 𝜎𝑧=1 when 𝑧=∞ --> Look at Sigmoid graph and equation and plug in inf,which yields 1

        TODO: 
            1. Adaptive Learning Rate
            2. Momentum 
            3. Label Smoothing
'''

def sigma(X,W):
    z = np.matmul(X,W.T)
    return (1/(1+np.exp(-z)))

class logistic_regression(object):
    def __init__(self):  
        self.W = None
    
    def fit(self,X,y,batch_size=512,lr=0.1, tol =0.001, max_it = 100, display_period=-1,lr_reduction=.25,patience=1, momentum=.9,label_smoothing=.95): 
        
        self.n_classes = np.amax(y)+1
        
        if display_period==-1:
            display_period = max_it+1
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        batches_per_epoch = X1.shape[0]//batch_size
        print('Batch Size: ', batch_size)
        print('Total amount of samples: ',X1.shape[0])
        print('Total batches per epoch: ',batches_per_epoch)
        self.acc_list, self.mse_list = [], []
        n, m = X1.shape                  # the training set has n examples and m attributes
        y_oh = one_hot(y)
        
        #Label smoothing - replace y_oh with y_smooth
        y_smooth = (y_oh * label_smoothing) + (1-label_smoothing)/self.n_classes
        self.W =  (np.random.random((y_smooth.shape[1],m))-0.5)/100    # w is kxm
        
        prevGradient = 0 #momentum - will hold the average gradient of previous 
        lowerBound = 0 #Used in adaptive learning - is the lower bound since the last update of lr
        for i in range(1,max_it+1):
            ind = np.random.permutation(X1.shape[0])%batches_per_epoch
            
            for b in range(batches_per_epoch):
                batch = (ind == b)
                S = sigma(X1[batch],self.W)
                Error = (S - y_smooth[batch])
                G = (Error*S*(1-S)).T
                #Momentum summary- mess with gradient so it can converge to min faster
                currGradient = np.matmul(G,X1[batch])/batch_size 
                currGradient = (currGradient * momentum) + (1-momentum)*prevGradient #prevGrad = avg grad estimate from previous batch
                prevGradient = np.mean(currGradient)
                self.W = self.W - lr*currGradient
                
            S = sigma(X1,self.W)
            Error = (S - y_smooth)
            pred = np.argmax(S,axis=1)
            acc = accuracy(y,pred)
            self.acc_list.append(acc)
            mse_train = np.mean(Error**2) 
            self.mse_list.append(mse_train)
            if mse_train<tol:
                break
            if i%display_period==0:  
                print('Epoch: {}/{}, mse = {:.6f}, accuracy = {:.6f}'.format(i,max_it,mse_train,acc))
                
            #Adpative Learning - update lr if mse has not changed since last patience epochs: i-lowerBound>1 indicates an interval since lr was last updated.
            if (i-lowerBound>1) and (i%patience == 0) and (mse_train >= np.min(self.mse_list[lowerBound:-1])):#take out most recent mse_list element since you want to compare the ones b4
                print('> Learning Rate changed when: epoch=',i,';lr=',lr, ';lr_red=',lr_reduction,';patience=',patience)
                print('> Proof: (', i,'%patience==0) and (mse_train=',mse_train,'>=',np.min(self.mse_list[:-1]), '= np.min(self.mse_list[:-1]))',)
                lr = lr*lr_reduction
                lowerBound=i
                print('> New lr = ', lr)
                
    def predict(self,X):
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        self.S = sigma(X1,self.W)
        pred = np.argmax(self.S,axis=1)
        return pred
    
if __name__ == "__main__":
    plt.close('all')
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise11\\'  # Use your own path here
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    model = logistic_regression()
    
    '''
    fit invokation:
            > Smaller batches allowed to see MSE fluctuate for adaptive learning
    '''
    model.fit(X_train,y_train,batch_size=8, lr=1, max_it = 20, display_period=1)
    
    pred_test =  model.predict(X_test)
    
    cm = confusion_matrix(y_test,pred_test)   
        
    print('Test MSE {:.6f}, Test accuracy: {:.6f}'.format(mse(model.S, one_hot(y_test)),accuracy(pred_test,y_test)))
    print(cm)
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(model.mse_list)
    ax[0].set_title('Training mean-squared error')
    ax[1].plot(model.acc_list)
    ax[1].set_title('Training accuracy')
    '''
Batch Size:  8
Total amount of samples:  63000
Total batches per epoch:  7875
Epoch: 1/20, mse = 0.016116, accuracy = 0.910032
Epoch: 2/20, mse = 0.014752, accuracy = 0.914841
Epoch: 3/20, mse = 0.014498, accuracy = 0.917063
Epoch: 4/20, mse = 0.014054, accuracy = 0.918127
Epoch: 5/20, mse = 0.013961, accuracy = 0.917571
Epoch: 6/20, mse = 0.013680, accuracy = 0.921492
Epoch: 7/20, mse = 0.013804, accuracy = 0.919778
> Learning Rate changed when: epoch= 7 ;lr= 1 ;lr_red= 0.25 ;patience= 1
> Proof: ( 7 %patience==0) and (mse_train= 0.013804431066295598 >= 0.01368018906516961 = np.min(self.mse_list[:-1]))
> New lr =  0.25
Epoch: 8/20, mse = 0.012845, accuracy = 0.924381
Epoch: 9/20, mse = 0.012827, accuracy = 0.923905
Epoch: 10/20, mse = 0.012871, accuracy = 0.924889
> Learning Rate changed when: epoch= 10 ;lr= 0.25 ;lr_red= 0.25 ;patience= 1
> Proof: ( 10 %patience==0) and (mse_train= 0.012871180017772783 >= 0.012827322209661318 = np.min(self.mse_list[:-1]))
> New lr =  0.0625
Epoch: 11/20, mse = 0.012698, accuracy = 0.924460
Epoch: 12/20, mse = 0.012671, accuracy = 0.925127
Epoch: 13/20, mse = 0.012657, accuracy = 0.925016
Epoch: 14/20, mse = 0.012633, accuracy = 0.925524
Epoch: 15/20, mse = 0.012637, accuracy = 0.925254
> Learning Rate changed when: epoch= 15 ;lr= 0.0625 ;lr_red= 0.25 ;patience= 1
> Proof: ( 15 %patience==0) and (mse_train= 0.012636965683615737 >= 0.012632584549591425 = np.min(self.mse_list[:-1]))
> New lr =  0.015625
Epoch: 16/20, mse = 0.012613, accuracy = 0.925349
Epoch: 17/20, mse = 0.012608, accuracy = 0.925571
Epoch: 18/20, mse = 0.012602, accuracy = 0.925365
Epoch: 19/20, mse = 0.012601, accuracy = 0.925460
Epoch: 20/20, mse = 0.012602, accuracy = 0.925413
> Learning Rate changed when: epoch= 20 ;lr= 0.015625 ;lr_red= 0.25 ;patience= 1
> Proof: ( 20 %patience==0) and (mse_train= 0.012601826823082135 >= 0.01260075772671086 = np.min(self.mse_list[:-1]))
> New lr =  0.00390625
Test MSE 0.015668, Test accuracy: 0.918286
[[656   0   1   1   0   1   6   0   4   1]
 [  1 793   2   2   0   4   1   0  10   1]
 [  8   6 663   6   8   2   9   9  28   3]
 [  6   3  29 636   1  19   3   7  16   8]
 [  2   1   1   1 638   0   6   0   6  18]
 [ 12   4   1  19  17 542  18   4  14   6]
 [  9   2   4   0   6   7 659   1   4   0]
 [  3   3  14   1   2   0   0 623   3  13]
 [  6   8  10  13   7  17   4   3 604   3]
 [  4   1   2   9  39   8   0  21   9 614]]
    '''