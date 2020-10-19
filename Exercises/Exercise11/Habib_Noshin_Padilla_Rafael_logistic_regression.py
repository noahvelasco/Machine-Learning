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
    
    def fit(self,X,y,batch_size=512,lr=0.1, tol =0.001, max_it = 100, display_period=-1,lr_reduction=.25,patience=3, momentum=.9,label_smoothing=.95): 
        
        self.n_classes = np.amax(y)+1
        
        if display_period==-1:
            display_period = max_it+1
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        batches_per_epoch = X1.shape[0]//batch_size
        print('Total amount of samples: ',X1.shape[0])
        print('Total batches per epoch: ',batches_per_epoch)
        self.acc_list, self.mse_list = [], []
        n, m = X1.shape                  # the training set has n examples and m attributes
        y_oh = one_hot(y)
        
        #Label smoothing - replace y_oh with y_smooth
        y_smooth = (y_oh * label_smoothing) + (1-label_smoothing)/self.n_classes
        self.W =  (np.random.random((y_smooth.shape[1],m))-0.5)/100    # w is kxm
        
        prevGradient = 0 #momentum - will hold the average gradient of previous batch
        for i in range(1,max_it+1):
            ind = np.random.permutation(X1.shape[0])%batches_per_epoch
            
            for b in range(batches_per_epoch):
                batch = (ind == b)
                S = sigma(X1[batch],self.W)
                Error = (S - y_smooth[batch])
                G = (Error*S*(1-S)).T
                #Momentum - mess with gradient so it can converge to min faster
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
           
            #Adpative Learning - update lr if mse has not changed since last patience epochs: TIP: Use a very small batch size = 20 in main
            if (i%patience == 0) and (mse_train >= np.min(self.mse_list[:-1])):#take out most recent mse_list element since you want to compare the ones b4
                print('> Learning Rate changed when: epoch=',i,';lr=',lr, ';lr_red=',lr_reduction,';patience=',patience)
                print('> Proof: (', i,'%patience==0) and (mse_train=',mse_train,'>=',np.min(self.mse_list[:-1]), '= np.min(self.mse_list[:-1]))',)
                lr = lr*lr_reduction
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
    model.fit(X_train,y_train,batch_size=10, lr=.2, max_it = 20, display_period=1)
    
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
Total amount of samples:  63000
Total batches per epoch:  6300
Epoch: 1/20, mse = 0.017624, accuracy = 0.902175
Epoch: 2/20, mse = 0.015983, accuracy = 0.909667
Epoch: 3/20, mse = 0.015386, accuracy = 0.912540
Epoch: 4/20, mse = 0.014898, accuracy = 0.914889
Epoch: 5/20, mse = 0.014673, accuracy = 0.916508
Epoch: 6/20, mse = 0.014369, accuracy = 0.917952
Epoch: 7/20, mse = 0.014135, accuracy = 0.917698
Epoch: 8/20, mse = 0.014020, accuracy = 0.919000
Epoch: 9/20, mse = 0.013902, accuracy = 0.920190
Epoch: 10/20, mse = 0.013771, accuracy = 0.920952
Epoch: 11/20, mse = 0.013755, accuracy = 0.920556
Epoch: 12/20, mse = 0.013713, accuracy = 0.921762
Epoch: 13/20, mse = 0.013789, accuracy = 0.921016
Epoch: 14/20, mse = 0.013484, accuracy = 0.922365
Epoch: 15/20, mse = 0.013530, accuracy = 0.922095
> Learning Rate changed when: epoch= 15 ;lr= 0.2 ;lr_red= 0.25 ;patience= 3
> Proof: ( 15 %patience==0) and (mse_train= 0.013529812382739247 >= 0.01348361360301765 = np.min(self.mse_list[:-1]))
> New lr =  0.05
Epoch: 16/20, mse = 0.013335, accuracy = 0.923286
Epoch: 17/20, mse = 0.013330, accuracy = 0.923540
Epoch: 18/20, mse = 0.013290, accuracy = 0.923476
Epoch: 19/20, mse = 0.013280, accuracy = 0.923683
Epoch: 20/20, mse = 0.013284, accuracy = 0.923651
Test MSE 0.015997, Test accuracy: 0.917286
[[655   0   1   1   0   1   5   0   6   1]
 [  1 790   1   3   0   4   1   0  13   1]
 [  7   6 657   7   9   2  11  11  29   3]
 [  6   5  22 639   2  18   2   7  19   8]
 [  2   0   1   1 635   0   7   0   5  22]
 [ 11   4   0  21  16 536  18   5  20   6]
 [ 10   2   4   0   6   6 660   0   4   0]
 [  3   3  13   1   1   0   1 624   3  13]
 [  5  10   6  13   6  16   5   2 605   7]
 [  4   1   3   9  32   8   0  21   9 620]]
    '''