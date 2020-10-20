import numpy as np
from utils import *
import math 
import time
from sklearn.neural_network import MLPClassifier
    
if __name__ == "__main__":  
    plt.close('all')
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise12\\'  # Use your own path here
    
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)/255
    y = np.load(data_path+'mnist_y.npy')
        
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50), verbose=True, random_state=1)
    
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    print('Training iterations  {} '.format(model.n_iter_))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy: {0:.6f}'.format(accuracy(y_test,pred)))
          
    cm = confusion_matrix(y_test,pred)   
    print(cm)
    
'''
Iteration 1, loss = 0.42505910
Iteration 2, loss = 0.19278103
Iteration 3, loss = 0.14250189
Iteration 4, loss = 0.11370394
Iteration 5, loss = 0.09422854
Iteration 6, loss = 0.08090426
Iteration 7, loss = 0.06966117
Iteration 8, loss = 0.05971882
Iteration 9, loss = 0.05354628
Iteration 10, loss = 0.04647668
Iteration 11, loss = 0.04083669
Iteration 12, loss = 0.03720972
Iteration 13, loss = 0.03216608
Iteration 14, loss = 0.02872423
Iteration 15, loss = 0.02533279
Iteration 16, loss = 0.02277929
Iteration 17, loss = 0.02066157
Iteration 18, loss = 0.01780601
Iteration 19, loss = 0.01678410
Iteration 20, loss = 0.01428919
Iteration 21, loss = 0.01321831
Iteration 22, loss = 0.01115138
Iteration 23, loss = 0.01032013
Iteration 24, loss = 0.00917719
Iteration 25, loss = 0.00832722
Iteration 26, loss = 0.00747989
Iteration 27, loss = 0.00667056
Iteration 28, loss = 0.00561173
Iteration 29, loss = 0.00492070
Iteration 30, loss = 0.00437070
Iteration 31, loss = 0.00551129
Iteration 32, loss = 0.00382419
Iteration 33, loss = 0.00313858
Iteration 34, loss = 0.00264688
Iteration 35, loss = 0.00362294
Iteration 36, loss = 0.00416501
Iteration 37, loss = 0.00298334
Iteration 38, loss = 0.00158934
Iteration 39, loss = 0.00125969
Iteration 40, loss = 0.00110873
Iteration 41, loss = 0.00112168
Iteration 42, loss = 0.00096303
Iteration 43, loss = 0.00110079
Iteration 44, loss = 0.01481780
Iteration 45, loss = 0.00182979
Iteration 46, loss = 0.00084808
Iteration 47, loss = 0.00070302
Iteration 48, loss = 0.00063492
Iteration 49, loss = 0.00058828
Iteration 50, loss = 0.00054470
Iteration 51, loss = 0.00052545
Iteration 52, loss = 0.00048338
Iteration 53, loss = 0.00242112
Iteration 54, loss = 0.01167613
Iteration 55, loss = 0.00166151
Iteration 56, loss = 0.00062133
Iteration 57, loss = 0.00048585
Iteration 58, loss = 0.00043105
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
Elapsed_time training  81.275778 
Training iterations  58 
Elapsed_time testing  0.035982 
Accuracy: 0.978857
[[664   0   1   0   1   2   1   0   1   0]
 [  1 807   1   0   0   0   2   1   2   0]
 [  2   3 722   2   1   0   3   3   5   1]
 [  0   1   8 701   0   4   0   2   4   8]
 [  0   1   1   0 665   0   0   2   0   4]
 [  0   0   1   3   1 621   6   0   2   3]
 [  2   3   1   0   4   0 680   0   2   0]
 [  1   1   8   0   0   0   0 645   2   5]
 [  1   2   3   1   2   1   2   2 658   3]
 [  0   0   2   3   6   3   0   3   1 689]]
   '''