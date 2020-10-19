import numpy as np
from utils import *
import math 

def sigma(X,W):
    z = np.matmul(X,W.T)
    return (1/(1+np.exp(-z)))

class logistic_regression(object):
    def __init__(self):  
        self.W = None
        self.n_classes = None
    
    def fit(self,X,y,batch_size=10,lr=0.1, tol =0.001, max_it = 100, display_period=-1, label_smoothing=1, momentum=0, lr_reduction=1, patience=1): 
        self.n_classes = np.amax(y)+1
        if display_period==-1:
            display_period = max_it+1
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        batches_per_epoch = X1.shape[0]//batch_size
        self.acc_list, self.mse_list = [], []
        n, m = X1.shape                  # the training set has n examples and m attributes
        y_oh = one_hot(y)
        gradient=0
        index = patience +2
        y_smooth = y_oh * label_smoothing + (1-label_smoothing)/self.n_classes
        self.W =  (np.random.random((y_smooth.shape[1],m))-0.5)/100    # w is kxm
        for i in range(1,max_it+1):
            ind = np.random.permutation(X1.shape[0])%batches_per_epoch
            for b in range(batches_per_epoch):
                batch = (ind == b)
                S = sigma(X1[batch],self.W)          
                Error = (S - y_smooth[batch])         
                G = (Error*S*(1-S)).T
                gradient_estimate = np.matmul(G,X1[batch])/batch_size  
                gradient = gradient * momentum + (1-momentum)*(gradient_estimate)
                self.W = self.W - lr*gradient
            S = sigma(X1,self.W)          
            Error = (S - y_smooth)       
            pred = np.argmax(S,axis=1)
            acc = accuracy(y,pred)
            self.acc_list.append(acc)
            mse_train = np.mean(Error**2) 
            self.mse_list.append(mse_train)
            if(i == index) :
                index += 4
                if round(self.mse_list[i-5],6) <= round(self.mse_list[i-1],6):
                    lr = lr*lr_reduction
                    print("Changed lr changed because", round(self.mse_list[i-5],6), " is smaller than ", round(self.mse_list[i-1],6))
            if mse_train<tol:
                break
            if i%display_period==0:  
                print('Epoch: {}/{}, mse = {:.6f}, accuracy = {:.6f}'.format(i,max_it,mse_train,acc))
           
    def predict(self,X):
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        self.S = sigma(X1,self.W)
        pred = np.argmax(self.S,axis=1)        
        return pred  
    
if __name__ == "__main__":  
    plt.close('all')
    
    data_path = 'C:\\Users\\noshin\\Desktop\\MachineLearning\\'  # Use your own path here
    
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    model = logistic_regression()
    
    model.fit(X_train,y_train, lr=10, max_it = 20, display_period=1, label_smoothing=1,momentum=0,lr_reduction=0.5,patience=3)
    
    pred_test =  model.predict(X_test)
    
    cm = confusion_matrix(y_test,pred_test)   
        
    print('Test MSE {:.6f}, Test accuracy: {:.6f}'.format(mse(model.S, one_hot(y_test)),accuracy(pred_test,y_test)))
    print(cm)
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(model.mse_list)
    ax[0].set_title('Training mean-squared error')
    ax[1].plot(model.acc_list)
    ax[1].set_title('Training accuracy')