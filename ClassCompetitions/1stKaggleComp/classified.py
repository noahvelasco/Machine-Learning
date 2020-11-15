import numpy as np
from utils import *
import math 
import time
from sklearn.neural_network import MLPRegressor
import re
    
if __name__ == "__main__":  
    #Data
    #data_path = '..//'  # Use your own path here    
    #X = np.load(data_path+'x_train.npy')
    #y = np.load(data_path+'y_train.npy').reshape(240000,)

    
    
    #Test Case Domains
    param_1 = [1, 5, 10, 15, 20]
    param_2 = [100, 500, 1000, 1500, 2000]
    param_3 = [32, 64, 128, 256, 512]
    param_4 = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    param_5 = [100, 200, 300, 400, 500]
    #X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    X_train = np.load('x_train.npy').astype(np.float32)
    X_test = np.load('x_test.npy').astype(np.float32)
    y_train = np.load('y_train.npy').astype(np.float32)
    
    Accuracies = []
    testCases = [
                [1,1,2,3,4],
                [1,2,3,4,5],
                [1,3,4,5,1],
                [1,4,5,1,2],
                [1,5,1,2,3],
                [2,1,3,5,2],
                [2,2,4,1,3],
                [2,3,5,2,4],
                [2,4,1,3,5],
                [2,5,2,4,1],
                [3,1,4,2,5],
                [3,2,5,3,1],
                [3,3,1,4,2],
                [3,4,2,5,3],
                [3,5,3,1,4],
                [4,1,5,4,3],
                [4,2,1,5,4],
                [4,3,2,1,5],
                [4,4,3,2,1],
                [4,5,4,3,2],
                [5,1,1,1,1],
                [5,2,2,2,2],
                [5,3,3,3,3],
                [5,4,4,4,4],
                [5,5,5,5,5]
                            ]
    
    m = 1
    with open("modelSummaries.txt", mode="w", encoding="utf-8") as model_summaries:
        for case in testCases:
            hidden_layers = param_1[int(case[0])-1]
            number_nodes = param_2[int(case[1])-1]
            batch_sizes = param_3[int(case[2])-1]
            lr = param_4[int(case[3])-1]
            max_iters = param_5[int(case[4])-1]
            param_1_2 = ()
            number_nodes = (number_nodes,)
            for i in range(hidden_layers):
                param_1_2 = param_1_2 + number_nodes
            print('tc ', m)
            model_summaries.write('tc ' + str(m))
            model_summaries.write('\n')
            print("Number of hidden layers and nodes per layer = ", param_1_2)
            model_summaries.write("Number of hidden layers and nodes per layer = " + str(param_1_2))
            model_summaries.write('\n')
            print("batch sizes = ", batch_sizes)
            model_summaries.write("batch sizes = " + str(batch_sizes))
            model_summaries.write('\n')
            print('learning rate = ', lr)
            model_summaries.write('learning rate = ' + str(lr))
            model_summaries.write('\n')
            print('maximum iterations = ', max_iters)
            model_summaries.write('maximum iterations = ' + str(max_iters))
            model_summaries.write('\n')
            model = MLPRegressor(solver='adam', max_iter = max_iters, learning_rate_init = lr, alpha=1e-5, batch_size = batch_sizes, hidden_layer_sizes= param_1_2, verbose=True, random_state=1)
            print('begin training model ', m)
            start = time.time()
            model.fit(X_train, y_train.ravel())
            elapsed_time = time.time()-start
            print('Elapsed_time training  {0:.6f} '.format(elapsed_time))
            model_summaries.write('Elapsed_time training: ' + str(elapsed_time))
            model_summaries.write('\n')
            print('Training iterations  {} '.format(model.n_iter_))  
            model_summaries.write('Training iterations: ' + str(model.n_iter_))
            model_summaries.write('\n')
            start = time.time()       
            pred = model.predict(X_test)
            elapsed_time = time.time()-start
            #acc = mse(y_test, pred)
            print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))  
            model_summaries.write('Elapsed_time testing ' + str(elapsed_time))
            model_summaries.write('\n')
            #print('MSE: {0:.6f}'.format(acc))
            #model_summaries.write('MSE: ' + str(acc))
            model_summaries.write('\n')
            model_summaries.write('\n')
            m += 1

