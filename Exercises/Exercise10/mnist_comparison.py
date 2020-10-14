import numpy as np
import time
import naive_bayes as nb
from utils import *

# import logistic_regression as log_reg # Uncomment when code is ready
if __name__ == "__main__":  
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exams\\Exam1\\'  # Use your own path here
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    plt.close('all')
    model = nb.naive_bayes()
    #model = log_reg.logistic_regression() # Replace previous line by this one
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
Change 1 line and the program should work
'''