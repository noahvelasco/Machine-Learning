import numpy as np
import matplotlib.pyplot as plt
import time 

"""
To understand precision and recall go to the link below:
    https://www.bmc.com/blogs/confusion-precision-recall/
"""



#(Total classification matches)/(Total Classifications)
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

'''
    confusion_matrix() - a len(TrueLabel) x len(PredictedLabel) matrix
                        called cm[i,j] that shows how many time class i was
                        classified as class j. Ex if cm[4,9]=13 then that means
                        class 4 was mistaken as a 9 13 times.
                        
                        Algorithm:
                            1. Iterate through the y_true and y_pred arrays entirely.
                                i. Increment the value by one in cm[i][j] where i and j 
                                    are the true and pred corresponding positions values
                        
'''
def confusion_matrix(y_true,y_pred):
    
    cm = np.zeros((np.amax(y_true)+1,np.amax(y_true)+1),dtype=int)
    
    for i in range(len(y_true)):
        
        cm[y_true[i]][y_pred[i]] += 1
        
    return cm

'''
    precision() - Defined as: (also called positive predictive value) is total amount
                you predicted correct(True Positive) out of the total amount
                you predicted correct plus total amount you thought you predicted correct (False Positive)
                        
                Precision = (True Positive)/ (True Positive + False Positive)
                false positive = a value predicted as positive but is actually negative
                               = a value predicted as 'positive_class' but is actually not that class
                
                Goal - Calculate the precision of each classification (0-9)
                
                Algorithm: 
                    1. For each classification go through both arrays(10 times^)
                        i. Keep count of True Positives for each classification 
                            and total False positives
                    2. Plug true and false positives into the formula

'''
def precision(y_true,y_pred,positive_class):
    
    true_pos = 0
    false_pos = 0
    
    #Could've used y_pred-doesnt matter since same size
    for i in range(len(y_true)):
        
        #if true positive = if actual and pred match
        if ( positive_class == y_true[i] ) and (y_true[i] == y_pred[i]):
            true_pos += 1
        
        #if false positive = if predicted is positive_class but actually is not positive_class
        if (y_pred[i] == positive_class) and (y_true[i] != y_pred[i]):
            false_pos +=1
   
    return true_pos / (true_pos + false_pos)



'''
    recall() - Defined as: (also known as sensitivity -wikipedia) is the total 
                amount of matches predicted right(True Positive) out of 
                total matches predicted right + total matches predicted wrong(false negatives). 
                Its used to measure total amount of relevant items[wikipedia].
                                
                Recall = (True Positive) / (True Positive + False Negative)
                
                Goal - Calculate the recall of each classification(0-9)
                
                Algorithm(Similar to precision): 
                    1. For each classification go through both arrays(10 times^)
                        i. Keep count of total True Positives for each 
                            classification and total False negatives
                    2. Plug true and false positives into the formula
'''
def recall(y_true,y_pred,positive_class):
    
    true_pos = 0
    false_neg = 0
    
    #Could've used y_pred-doesnt matter since same size
    for i in range(len(y_true)):
        
        #if true positive = if actual and pred match
        if ( positive_class == y_true[i] ) and (y_true[i] == y_pred[i]):
            true_pos += 1
        
        #if false negative = if value should be positive_class but is predicted to not be positive class
        if (y_true[i] == positive_class) and (y_true[i] != y_pred[i]):
            false_neg +=1 
    
    rec = true_pos / (true_pos + false_neg)
    
    return rec



if __name__ == '__main__':
   
    
    print('\nEvaluating Algorithm 1')
    y_test_a1 =  np.load('y_test_a1.npy')
    pred_a1 =  np.load('pred_a1.npy')
    print('Accuracy:  {:.4}'.format(accuracy(y_test_a1,pred_a1)))
    print('Confusion matrix:')
    print(confusion_matrix(y_test_a1,pred_a1))
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
    
    