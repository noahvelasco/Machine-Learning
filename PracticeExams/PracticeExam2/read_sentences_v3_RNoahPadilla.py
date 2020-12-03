'''
Author: Olac Fuentes
Modifications: R Noah Padilla

 --->>>>> OUTPUT IS COMMENTED AT BOTTOM <<<<<---

Practice Exam 2:

    (a) Compute a representation where each sentence is represented by the average embedding in the sentence (thus
        your dataset will be of size (12245,50)). Notice that if a particular row in a sentence representation contains
    all zeros, it means that it was added as padding, thus it should not be included in the average.
    (b) Split the data into training and testing set.
    (c) Compare the performance of multilayer perceptron, decision tree, and random forests to classify the dataset.
    (d) Compress the data to only 5 components using Principal Component Analysis
    (e) Compare the performance of multilayer perceptron, decision tree, and random forests to classify the reduced
        dataset.
'''
import bs4 as bs
import urllib.request
import numpy as np
import time
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

def avg_emb(X):
    #New reduced version of X whos shape should be (12245, 50)
    newX = np.zeros((X.shape[0],X.shape[2]))
    for i,sent in enumerate(X):
        #take the average of the columns excluding 0s
        newX[i] = np.sum(sent,axis=0)/np.count_nonzero(sent, axis=0)
    return newX

def read_embeddings(n=1000):
    # Reads n embeddings from file
    # Returns a dictionary were embedding[w] is the embeding of string w
    embedding = {}
    count = 0
    with open('glove.6B.50d.txt', encoding="utf8") as f: 
        for line in f: 
            count+=1
            ls = line.split(" ")
            emb = [np.float32(x) for x in ls[1:]]
            embedding[ls[0]]=np.array(emb)
            if count>= n:
                break
    return embedding

def sentence_embeddings(s,emb,n=20):
    se = np.zeros((n,50))
    for i in range(min(len(s),n)):
        try:
            se[i] = emb[s[i]]
        except: # Set all embeding values to -1 if word is unknown
           pass
    return se

def get_words(st):
    st = st.lower()
    st = st.replace('\r\n', ' ')
    st = ''.join( c for c in st if  c in lowercase)
    words = st.split()
    return words

def get_sentence_list(url):
    paragraphs = []
    word_lists = []
    sentence_list = []
    data = urllib.request.urlopen(url).read()
    soup = bs.BeautifulSoup(data,'lxml')
    count = 0
    for paragraph in soup.find_all('p'):
        par  = paragraph.string
        if par:
            par = par.replace('\r\n', ' ')
            sent = par.split('.')
            for s in sent:
                sentence_list.append(s+'.')
                words = get_words(s)
                if len(words)>0:
                    word_lists.append(words)
    return word_lists

if __name__ == "__main__":  
    url_list = ['http://www.gutenberg.org/files/215/215-h/215-h.htm', 'http://www.gutenberg.org/files/345/345-h/345-h.htm', 'http://www.gutenberg.org/files/1661/1661-h/1661-h.htm']
    lowercase = ''.join(chr(i) for i in range(97,123)) + ' '
    sentence_list = []       
    for u, url in enumerate(url_list):
        word_lists = get_sentence_list(url)
        print('Book {} contains {} sentences'.format(u,len(word_lists)))
        lengths = np.array([len(wl) for wl in word_lists])
        print('Sentence length stats:')
        print('min = {} max = {} mean = {:4f}'.format(np.min(lengths),np.max(lengths),np.mean(lengths)))
        sentence_list.append(word_lists)
        
    vocabulary_size = 100000
    embedding = read_embeddings(vocabulary_size)
    
    X,y  = [],[]
    empty = 0
    for i, sentences in enumerate(sentence_list):
        for s in sentences:
            se = sentence_embeddings(s,embedding,n=20)
            if np.sum(np.abs(se))>0:
                X.append(sentence_embeddings(s,embedding,n=125))
                y.append(i)
            else:
                print('empty sentence in book',i)
                empty+=1
    
    X = np.array(X)
    y = np.array(y)
    print("X shape",X.shape)
    
    print(empty,'empty sentences found')
    np.save('X_books.npy',X)
    np.save('y_books.npy',y)
    
    #------------------------------ PART A ------------------------------
    reducedX = avg_emb(X)
    print("Reduced X shape",reducedX.shape)
    
    #------------------------------ PART B ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(reducedX, y)
    
    #------------------------------ PART C ------------------------------
    
    #MLP
    print('-- MLP --')
    model = MLPClassifier(solver='adam', alpha=1e-5, batch_size = 100 ,learning_rate='adaptive',momentum=0.95,  hidden_layer_sizes=(400), random_state=1)
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
    
    #DECISION TREE
    print('-- DTREE --')
    model = DecisionTreeClassifier(criterion='entropy', max_depth=30, max_leaf_nodes=5082 , random_state=0)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy: {0:.6f}'.format(accuracy(y_test,pred)))
    
    #RANDOM FORESTS
    print('-- RANDOM FORESTS --')
    model = RandomForestClassifier(n_estimators=200,random_state=0)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy: {0:.6f}'.format(accuracy(y_test,pred)))
    
    #------------------------------ PART D ------------------------------
    print("** PCA TRANSFORMATION **")
    pca = PCA(n_components=5)
    pca.fit(X_train)
    ev = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)
    cum_ev = cum_ev/cum_ev[-1]
    
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    #------------------------------ PART E ------------------------------
    #MLP
    print('-- MLP --')
    model = MLPClassifier(solver='adam', alpha=1e-5, batch_size = 100 ,learning_rate='adaptive',momentum=0.95,  hidden_layer_sizes=(400), random_state=1)
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
    
    #DECISION TREE
    print('-- DTREE --')
    model = DecisionTreeClassifier(criterion='entropy', max_depth=30, max_leaf_nodes=5082 , random_state=0)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy: {0:.6f}'.format(accuracy(y_test,pred)))
    
    #RANDOM FORESTS
    print('-- RANDOM FORESTS --')
    model = RandomForestClassifier(n_estimators=200,random_state=0)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy: {0:.6f}'.format(accuracy(y_test,pred)))
    
    '''
    Book 0 contains 1618 sentences
    Sentence length stats:
    min = 1 max = 122 mean = 19.241656
    Book 1 contains 4219 sentences
    Sentence length stats:
    min = 1 max = 125 mean = 17.345105
    Book 2 contains 6423 sentences
    Sentence length stats:
    min = 1 max = 101 mean = 15.205511
    empty sentence in book 1
    empty sentence in book 1
    empty sentence in book 1
    empty sentence in book 1
    empty sentence in book 1
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    empty sentence in book 2
    X shape (12245, 125, 50)
    15 empty sentences found
    Reduced X shape (12245, 50)
    -- MLP --
    Elapsed_time training  35.314727 
    Training iterations  200 
    Elapsed_time testing  0.013995 
    Accuracy: 0.672110
    -- DTREE --
    Elapsed_time training  1.345994 
    Elapsed_time testing  0.003001 
    Accuracy: 0.510777
    -- RANDOM FORESTS --
    Elapsed_time training  12.401069 
    Elapsed_time testing  0.163058 
    Accuracy: 0.633899
    ** PCA TRANSFORMATION **
    -- MLP --
    Elapsed_time training  25.228817 
    Training iterations  200 
    Elapsed_time testing  0.012996 
    Accuracy: 0.538210
    -- DTREE --
    Elapsed_time training  0.210899 
    Elapsed_time testing  0.001000 
    Accuracy: 0.466035
    -- RANDOM FORESTS --
    Elapsed_time training  4.625664 
    Elapsed_time testing  0.148902 
    Accuracy: 0.513063
    '''