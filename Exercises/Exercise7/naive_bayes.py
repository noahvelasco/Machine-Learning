import math
import numpy as np
import matplotlib.pyplot as plt
import time 
import sys

class naive_bayes():
    def __init__(self):  
        self.n_classes = None
        self.p_att_given_class = None
        
    def fit(self,X,y): 
        # Assumes X is binary for predict1 and predict2
        self.n_classes = np.amax(y)+1
        self.p_class = np.zeros(self.n_classes)
        self.p_att_given_class = np.zeros((self.n_classes,X.shape[1]))
        self.means = np.zeros((self.n_classes,X.shape[1]))#10 classes and calculate mean for each attribute
        self.stds = np.zeros((self.n_classes,X.shape[1]))#10 classes and calculate std for each attribute
        
        for i in range(self.n_classes):
            self.p_class[i] = np.sum(y==i)/len(y)
            self.p_att_given_class[i] = np.mean(X[y==i],axis=0)
            self.means[i] = np.mean(X[y==i],axis=0) + 1e-200 #smoothing
            self.stds[i] = np.std(X[y==i],axis=0) + 1e-200 #smoothing
    
    
    #For num 1 on exercise
    def predict1(self,x_test):
        pred =  np.zeros(x_test.shape[0],dtype=int)
        probs = np.zeros((x_test.shape[0],self.n_classes))
        for i,x in enumerate(x_test):
            p = self.p_att_given_class*x + (1-self.p_att_given_class)*(1-x)
            m = np.prod(p,axis=1)
            probs[i] = m*self.p_class
        probs = probs/np.sum(probs,axis=1).reshape(-1,1)
        pred = np.argmax(probs,axis=1)
        return pred,probs
    
    '''
    #^^^^^^^^^^^^
    Elapsed_time training:  0.179914 secs
    Elapsed_time testing: 0.605708 secs
    Accuracy: 0.840714 
    '''
    
    
    #For num 2 on exercise
    def predict2(self,x_test):
        pred =  np.zeros(x_test.shape[0],dtype=int) 
        probs = np.zeros((x_test.shape[0],self.n_classes))
        for i,x in enumerate(x_test):
            p = self.p_att_given_class*x + (1-self.p_att_given_class)*(1-x)
            m = np.prod(p,axis=1)
            probs[i] = m * self.p_class
        probs += 1e-200 #smoothing
        probs = probs/np.sum(probs,axis=1).reshape(-1,1)
        pred = np.argmax(np.log(probs),axis=1)#<<<<<<<<<Include the log and its the same as top predict
        return pred,probs
    
    '''
    Elapsed_time training:  0.179915 secs
    Elapsed_time testing: 0.572727 secs
    Accuracy: 0.834429
    '''

    #For num 3 on exercise - real-valued attributes
    def predict3(self,x_test):
        
        #x_test is using 0-255 values for each attribute
        
        pred =  np.zeros(x_test.shape[0],dtype=int) #(7000,784)
        #Go through every test point
        for i,x in enumerate(x_test):
            
            allprobs = np.zeros(self.n_classes)
            #For every test point generate 10 probs using formula and get max
            for j in range(self.n_classes):
                
                term1 = 1/(math.sqrt(2 * math.pi) * self.stds[j] )
                term2 = np.exp( -1 * (((x-self.means[j])**2 ) / ( self.stds[j]**2 )) )
                fw = np.multiply(term1 , term2)
                '''
                print("x.shape",x.shape)
                print("--------------T2")
                print(term1)
                print("--------------T2")
                print(term2)
                print("--------------fw")
            
                sys.exit(0)
                '''
                allprobs[j] = np.prod(fw) * self.p_class[j]
            #print(allprobs)
               
            #create probs and get argmax and add to pred[i]
            allprobs += 1e-200 #smoothing
            allprobs = allprobs/np.sum(allprobs)
            pred[i] = np.argmax(allprobs)         
            
        return pred
    
    '''
    <<< Part 3 of exercise >>>
    Elapsed_time training:  0.624700 secs
C:/Users/npizz/Desktop/Machine-Learning/Exercises/Exercise7/naive_bayes.py:79: RuntimeWarning: divide by zero encountered in true_divide
  term1 = 1/(math.sqrt(2 * math.pi) * self.stds[j] )
C:/Users/npizz/Desktop/Machine-Learning/Exercises/Exercise7/naive_bayes.py:80: RuntimeWarning: overflow encountered in square
  term2 = np.exp( -1 * (((x-self.means[j])**2 ) / ( self.stds[j]**2 )) )
C:/Users/npizz/Desktop/Machine-Learning/Exercises/Exercise7/naive_bayes.py:81: RuntimeWarning: invalid value encountered in multiply
  fw = np.multiply(term1 , term2)
Elapsed_time testing: 3.925118 secs
Accuracy: 0.097143 
    '''

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
    
    #------------------------#1 on exercise------------------------------------
    thr = 127.5
    X = (X>thr).astype(int)
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    print("<<< Part 1 of exercise >>>")
    model1 = naive_bayes()
    start = time.time()
    model1.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  
    plt.close('all')
    
    start = time.time()
    pred1,probs1 = model1.predict1(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred1,y_test)))
    #-----------------------#2 on exercise-------------------------------------
    print("<<< Part 2 of exercise >>>")
    model2 = naive_bayes()
    start = time.time()
    model2.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  

    plt.close('all')

    start = time.time()
    pred2, probs2 = model2.predict2(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred2,y_test)))   
    
    #-----------------------#3 on exercise-------------------------------------
    
    #used for predict3
    X = np.load('mnist_X.npy').astype(np.float32).reshape(-1,28*28) #use 0-255 values for this one
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    
    print("<<< Part 3 of exercise >>>")
    model3 = naive_bayes()
    start = time.time()
    model3.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  

    plt.close('all')
    start = time.time()
    pred3 = model3.predict3(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred3,y_test)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    