import numpy as np
import matplotlib.pyplot as plt
import time 

class naive_bayes():
    def __init__(self):  
        self.n_classes = None
        self.p_att_given_class = None
        
    def fit(self,X,y): 
        # Assumes X is binary
        self.n_classes = np.amax(y)+1
        self.p_class = np.zeros(self.n_classes)
        self.p_att_given_class = np.zeros((self.n_classes,X.shape[1]))
        for i in range(self.n_classes):
            self.p_class[i] = np.sum(y==i)/len(y)
            self.p_att_given_class[i] = np.mean(X[y==i],axis=0)
            
    def predict(self,x_test):
        pred =  np.zeros(x_test.shape[0],dtype=int)
        probs = np.zeros((x_test.shape[0],self.n_classes))
        for i,x in enumerate(x_test):
            p = self.p_att_given_class*x + (1-self.p_att_given_class)*(1-x)
            m = np.prod(p,axis=1)
            probs[i] = m*self.p_class
        probs = probs/np.sum(probs)
        pred = np.argmax(probs,axis=1)
        return pred,probs
   
def display_probabilities(P):
    fig, ax = plt.subplots(1,10,figsize=(10,1))
    for i in range(10):
        ax[i].imshow(P[i].reshape((28,28)),cmap='gray')
        ax[i].axis('off')    
    
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]
   
def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]    
    
if __name__ == "__main__":  
    plt.close('all')
   
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28)
    y = np.load('mnist_y.npy')
    
    thr = 127.5
    X = (X>thr).astype(int)
     
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    model = naive_bayes()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  
    
    plt.close('all')
    display_probabilities(model.p_att_given_class)
    
    start = time.time()       
    pred,probs = model.predict(X_test)
    print(">>> prs",probs.shape)
    print(">>> pds",pred.shape)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred,y_test)))
    
    
    
    