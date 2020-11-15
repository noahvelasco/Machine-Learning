import numpy as np
import time
import naive_bayes as nb
from utils import *
import logistic_regressionRNoahPadilla as log_reg

if __name__ == "__main__":
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise10\\'
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    plt.close('all')
    
    print('> Naive Bayes Model')
    model = nb.naive_bayes()
    start = time.time()
    model.fit(X_train, y_train) 
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred,y_test)))
   
    #-- Below code is same as above just using another model for better readibility---------------------
    
    print('> Logistic Regression Model')
    model = log_reg.logistic_regression()
    start = time.time()
    model.fit(X_train,y_train,batch_size= 512,lr=1, tol =0.001,display_period=1,max_it = 20) 
    elapsed_time = time.time()-start
    print('Elapsed_time training:  {0:.6f} secs'.format(elapsed_time))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing: {0:.6f} secs'.format(elapsed_time))   
    print('Accuracy: {0:.6f} '.format(accuracy(pred,y_test)))
    '''
> Naive Bayes Model
Elapsed_time training:  0.860587 secs
Elapsed_time testing: 1.223415 secs
Accuracy: 0.831714 
> Logistic Regression Model
Epoch: 1/20, mse = 0.029980, accuracy = 0.8668
Epoch: 2/20, mse = 0.025774, accuracy = 0.8821
Epoch: 3/20, mse = 0.024000, accuracy = 0.8879
Epoch: 4/20, mse = 0.022950, accuracy = 0.8915
Epoch: 5/20, mse = 0.022226, accuracy = 0.8939
Epoch: 6/20, mse = 0.021684, accuracy = 0.8959
Epoch: 7/20, mse = 0.021241, accuracy = 0.8974
Epoch: 8/20, mse = 0.020921, accuracy = 0.8987
Epoch: 9/20, mse = 0.020604, accuracy = 0.8996
Epoch: 10/20, mse = 0.020339, accuracy = 0.9004
Epoch: 11/20, mse = 0.020122, accuracy = 0.9012
Epoch: 12/20, mse = 0.019944, accuracy = 0.9025
Epoch: 13/20, mse = 0.019753, accuracy = 0.9031
Epoch: 14/20, mse = 0.019604, accuracy = 0.9035
Epoch: 15/20, mse = 0.019503, accuracy = 0.9047
Epoch: 16/20, mse = 0.019335, accuracy = 0.9047
Epoch: 17/20, mse = 0.019234, accuracy = 0.9051
Epoch: 18/20, mse = 0.019117, accuracy = 0.9061
Epoch: 19/20, mse = 0.019016, accuracy = 0.9061
Epoch: 20/20, mse = 0.018920, accuracy = 0.9070
Elapsed_time training:  11.468510 secs
Elapsed_time testing: 0.004997 secs
Accuracy: 0.901571
    '''