import numpy as np
import matplotlib.pyplot as plt
from utils import *
import time 

class naive_bayes():
    def __init__(self):
        self.n_classes = None
        self.mean_att_given_class = None
        self.var_att_given_class = None
        
    def fit(self,X,y): 
        self.n_classes = np.amax(y)+1
        self.p_class = np.zeros(self.n_classes)
        self.mean_att_given_class = np.zeros((self.n_classes,X.shape[1]))
        self.var_att_given_class = np.zeros((self.n_classes,X.shape[1]))
        for i in range(self.n_classes):
            self.p_class[i] = np.sum(y==i)/len(y)
            self.mean_att_given_class[i] = np.mean(X[y==i],axis=0)
            self.var_att_given_class[i] = np.var(X[y==i],axis=0)+.01 # Smoothing factor
            
    def predict(self,x_test):
        pred =  np.zeros(x_test.shape[0],dtype=int)
        probs =  np.zeros((x_test.shape[0],self.n_classes))
        
        for i,x in enumerate(x_test):
            d = self.mean_att_given_class-x
            d = d*d/self.var_att_given_class
            p = -d - np.log(2*np.pi*self.var_att_given_class)/2
            p = np.sum(p,axis=1)
            probs[i] = p+np.log(self.p_class)
            
        pred = np.argmax(probs,axis=1)
        return pred
   
if __name__ == "__main__":
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exams\\Exam1\\'  # Use your own path here
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    plt.close('all')
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