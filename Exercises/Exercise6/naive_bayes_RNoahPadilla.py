import numpy as np
import matplotlib.pyplot as plt
import time 
'''
Goal: Use the naive bayes classifier to classify test points given from the MNIST dataset
'''
class naive_bayes():
    def __init__(self):  
        self.n_classes = None
        self.p_att_given_class = None
      
    #write
    def fit(self,X,y): 
        #Passes xtrain and ytrain
        
        # Assumes X is binary
        self.n_classes = np.amax(y)+1
        
        self.p_class = np.zeros(self.n_classes)
        self.p_att_given_class = np.zeros((self.n_classes,X.shape[1]))

        #populate self.p_class with probabilities of each class
        for i in range(len(self.p_class)):
            self.p_class[i] = np.sum(y==i)/len(y)
        
        #populate self.p_att_given_class with probabilities of every attribute being 1 for every class; c =class, a=attribute
        for c in range(self.p_att_given_class.shape[0]):
            for a in range(self.p_att_given_class.shape[1]):
                
                self.p_att_given_class[c][a] = np.sum((y==c) * (X[:,a]==1))/np.sum(y==c)
                
    '''
    predict() - predicts the classification of  each test point from x_test 
                using the naive bayes theorems and stores it into 'pred'
                
                BEST TO UNDERSTAND IF ITS DRAWN OUT - look at MindMap.pdf for an example thats smail and similar 
                
                --------------Givens--------------
                self.p_class contains probabilities of each class
                self.p_att_given_class contains probabilites of each attribute being a 1(white)
                x_test = 7,000 samples that are 784 length and either contain 1's(pixel>127.5) or 0's (pixel<127.5)
                
                --------------What we need--------------
                pred[]- which will contain calssifications(0-9 values) of all the test points
                
                
                Algorithm using Naive Bayes:
                    1. Create 'pred[]' - will contain CLASSIFCATIONS(0-9 VALUES) of all test points from x_test in respective order; pred[0] = classification of x_test[0]
                    2. Start with a test point 'x_test[i]'
                    3. Create 'allProbs[]' - stores the probabilities of x_test[i] being each of the classifications (0-9); allProbs[0] = prob of x_test[i] being a 0 and so on
                    4. Get probability='prob' of x_test[i] being each of the 9 classes using N.B.T. and store them into allProbs[];
                        - N.B.T. -> prob = P(c_j) * P(a_1 | c_j) * P(a_2 | c_j) *, ... ,* P(a_len(self.p_att_given_class[j]) | c_j)
                    5. Get the highest probability index from allProbs[] and add to pred[i]
                    6. Repeat steps 2-5 for the rest of the test points
    '''
    
    def predict(self,x_test):
        
        #1. pred[] stores all classifications of x_test in respective order
        pred =  np.zeros(x_test.shape[0],dtype=int)
        
        #2. Go through every test point and figure out its CLASSIFICATION (indicated by index = j)
        for i in range(len(x_test)):
            
            #3. allProbs[] contains 10 probabilities for test point[i] and highest one calculated will be put into pred[i]
            allProbs = np.zeros(len(self.p_class))
            
            #4. Calculate prob of self.p_class[j] being x_test[i]; j = class we are at from 'self'
            for j in range(len(self.p_class)):
                
                #prob - the probability of self.p_class[j] being x_test[i]; updates in nested forloop and uses naive bayes theorem
                prob = self.p_class[j]
                
                #go through every att from self.p_att_given_class ( every att of class j ) and use bayes theorem; look at MindMap.pdf
                for k in range(len(self.p_att_given_class[j])):
                
                    #if value of x_test[i][k] == 1 then get prob of it being a 1 = self.p_class[j][k]
                    if x_test[i][k] == 1:
                        prob *= self.p_att_given_class[j][k]
                    
                    #if values of x_test[i][j] == 0 then get prob of being a 0 = (1-self.p_class[j][k])
                    if x_test[i][k] == 0:
                        prob *= (1-self.p_att_given_class[j][k])
                        
                allProbs[j] = prob
                               
            #5. save the index of the largest number=highest prob that classifies x_test[i]; index = classificiation in this case
            pred[i] = np.argmax(allProbs)
            
            #6 Repeat
            
        #return list of CLASSIFICATIONS of every test point from x_test in respective order
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
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred,y_test)))    
    
''' 
Fuentes Sample run output:
    
Elapsed_time training:  0.149601 secs
Elapsed_time testing: 0.349067 secs
Accuracy: 0.833429    

Noah Sample run output:
    
Elapsed_time training:  5.667286 secs
Elapsed_time testing: 117.237920 secs
Accuracy: 0.837000
'''    