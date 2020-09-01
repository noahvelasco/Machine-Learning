import numpy as np
import matplotlib.pyplot as plt
import time 


def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]
#Modify
def confusion_matrix(y_true,y_pred):
    cm = np.zeros((np.amax(y_true)+1,np.amax(y_true)+1),dtype=int)
    
    
    
    
    return cm

#Modify
def precision(y_true,y_pred,positive_class):
    return 0.0546543677

#Modify
def recall(y_true,y_pred,positive_class):
    return 0.06572223



if __name__ == '__main__':


   

    print('\nEvaluating Algorithm 1')
    y_test_a1 =  np.load('y_test_a1.npy')
    pred_a1 =  np.load('pred_a1.npy')
    print('Accuracy:  {:.4}'.format(accuracy(y_test_a1,pred_a1)))
    print('Confusion matrix:')
    print(confusion_matrix(y_test_a1,pred_a1))
    
    '''
    print('Precision and recall:')
    for i in range(np.amax(y_test_a1)+1):
        print('Positive class {}, precision: {:.4}, recall: {:.4}'.format(i, precision(y_test_a1,pred_a1,i),recall(y_test_a1,pred_a1,i)))
    
    print('\nEvaluating Algorithm 2')
    y_test_a2 =  np.load('y_test_a2.npy')
    pred_a2 =  np.load('pred_a2.npy')
    print('Accuracy:  {:.4}'.format(accuracy(y_test_a2,pred_a2)))
    print('Precision and recall:')
    for i in range(np.amax(y_test_a2)+1):
        print('Positive class {}, precision: {:.4}, recall: {:.4}'.format(i, precision(y_test_a2,pred_a2,i),recall(y_test_a2,pred_a2,i)))
    '''
    
    