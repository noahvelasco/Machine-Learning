import numpy as np
import matplotlib.pyplot as plt
import time 

class naive_bayes():
    def __init__(self):  
        self.n_classes = None
        self.p_att_given_class = None
      
    #write
    def fit(self,X,y): 
        #Passes xtrain and ytrain
        
        # Assumes X is binary
        self.n_classes = np.amax(y)+1
        
        # Write code to compute self.p_class and self.p_att_given_class as explained in the slides
        
        self.p_class = np.zeros(self.n_classes)
        self.p_class2 = np.zeros(self.n_classes)
        self.p_att_given_class = np.zeros((self.n_classes,X.shape[1]))
        self.p_att_given_class2 = np.zeros((self.n_classes,X.shape[1]))
        #populate self.p_class with probabilities of each class
        
        for i in range(len(self.p_class)):
            self.p_class[i] = np.sum(y==i)/len(y)
        '''
        #Bibek code
        self.p_class2 = np.asarray([np.sum(y==i)/len(y) for i in range(self.n_classes)])
        '''
        #----------------------------------------------
        
        #populate self.p_att_given_class with probabilities of every attribute; c =class, a=attribute
        for c in range(self.p_att_given_class.shape[0]):
            for a in range(self.p_att_given_class.shape[1]):
                
                self.p_att_given_class[c][a] = np.sum((y==c) * (X[:,a]==1))/np.sum(y==c)
                
        '''
        #Bibek code
        for i in range(self.n_classes):
            self.p_att_given_class2[i,:] = np.sum((y==i)*((X==1).T),axis=1)/np.sum(y==i)
        
        #compare mine vs Bibek results - yields true both same
        print(">>>>>", np.array_equal(self.p_class,self.p_class2) , " | ", np.array_equal(self.p_att_given_class,self.p_att_given_class2))
        '''
        
        
    #Predicts classifications of all x_test and returns them
    def predict(self,x_test):
        
        #7,000 samples that are 784 length
        
        pred =  np.zeros(x_test.shape[0],dtype=int)
        
        #Use Naives Bay Classifier to populate pred with <classifications>; will reset when using new test sample
        for i in range(len(x_test)):
                      
            allDistances = np.zeros(x_test.shape[0],dtype=int)
            for j in range(len(self.p_class)):
                
                #now use naive bayes theorem and add to allProbabilities
                allDistances[j] = self.p_class[j] + np.prod(self.p_att_given_class[j,:])
    
            pred[i] = np.argmax(allDistances[i])
            
        return pred
   
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
    X=  (X>thr).astype(int)
    
     
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    model = naive_bayes()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  
    
    
    start = time.time()       
    pred = model.predict(X_test)
    #pred = np.zeros(X_test.shape[0],dtype=int)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred,y_test)))  
    
    display_probabilities(model.p_att_given_class)
    
    
    
''' 
Sample run output:
    
Elapsed_time training:  0.149601 secs
Elapsed_time testing: 0.349067 secs
Accuracy: 0.833429    
'''    