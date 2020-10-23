import numpy as np
from utils import *
import math 
import time
from sklearn.tree import DecisionTreeClassifier
 
'''
optimize to get over 90% acc - read documentation and optimize it
'''
if __name__ == "__main__":  
    plt.close('all')
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise13\\'
    
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)
    
    y = np.load(data_path+'mnist_y.npy')
    thr = 127
    X = (X>thr).astype(int)
    
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    model = DecisionTreeClassifier(random_state=0)
    
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} secs'.format(elapsed_time))  
    
    pred = model.predict(X_train)
    print('Accuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
         
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
