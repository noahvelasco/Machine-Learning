import numpy as np
from utils import *
import math 
import time
from sklearn.tree import DecisionTreeClassifier
 

if __name__ == "__main__":  
    
    print('ENSEMBLE WITH RANDOM ATTRIBUTE SELECTION')
    plt.close('all')
    
    data_path = 'C:\\Users\\npizz\\Desktop\\Machine-Learning\\Exercises\\Exercise15\\'  # Use your own path here
    
    X = np.load(data_path+'mnist_X.npy').reshape(-1,28*28)
    y = np.load(data_path+'mnist_y.npy')
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    ensemble = []
    att_list = []
    ensemble_size = 15
    start = time.time()
    for i in range(ensemble_size):
        print('Training ensemble model',i)
        model = DecisionTreeClassifier(max_depth=12)
        a = np.random.permutation(X.shape[1])[:X.shape[1]//2]
        model.fit(X_train[:,a], y_train)
        ensemble.append(model)
        att_list.append(a)
        elapsed_time = time.time()-start
        print('Elapsed time training so far {:.6f} secs'.format(elapsed_time))  
    

    start = time.time()
    votes = np.zeros((X_test.shape[0],np.amax(y)+1),dtype=int)
    for i, model in enumerate(ensemble):
        pred = model.predict(X_test[:,att_list[i]])
        print('Model {} accuracy: {:.6f}'.format(i,accuracy(y_test,pred)))
        for j in range(len(y_test)):
            votes[j,pred[j]] += 1
            
    ens_pred = np.argmax(votes,axis =1)    
    elapsed_time = time.time()-start
    print('Ensemble accuracy: {:.6f}'.format(accuracy(y_test,ens_pred)))
    print('Elapsed time testing {:.6f} secs'.format(elapsed_time))   