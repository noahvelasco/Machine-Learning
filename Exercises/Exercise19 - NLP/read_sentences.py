'''
Author: Olac Fuentes
Modified by: R Noah Padilla

THIS FILE IS USED TO VIEW VARIABLE ATTRIBUTES IN SPYDER - BASICALLY A TESTING FILE FOR "NLP_RNoahPadilla.ipynb"

The goal of this assignment is written below.

The program read_sentences.py reads sentences form online classic books and 
converts them to a list of sentences, where each sentence is a list of words.
    


    [x]1. Write a function that receives a sentence and returns a 2D array containing 
        the embeddings of the words in the sentence. Your function should receive the embeddings 
        dictionary, the sentence and the desired length of the representation; if the 
        sentence is shorter than the desired length, path the array with zeros; if it’s longer, 
        truncate the representation.
    
    [x]2. Apply the function to produce an embedding representation of each of the 
        sentences in the three books used in the read_sentences.py program and generate
        a dataset containing examples of 3 classes, one for each book.
        
                > apply function to each sentence from each book and save all of them into a data set
    
    []3. Randomly split the data into training and testing.
    
    []4. Train and test a system to determine the book each sentence belongs to.
    
    CLASSIFICATION PROBLEM

'''

from sklearn.model_selection import train_test_split
import bs4 as bs
import urllib.request
import numpy as np
import sys

'''
TODO
sent_embedder():
        - receives a sentence, word embeddings dictionary, and the desired length of the sentence represention
        - returns a 2D array containing each of the words embeddings for a given sentence| row = word and columns are the embedding values

'''
def sent_embedder(sent, emb, desLen):
    
    #>>> Trim/Truncate sentences based on 'uSeRs' desires
    if len(sent) < desLen:
        extraZeros = [0]*(desLen - len(sent))#add zeros instead
        extraZeros = [str(x) for x in extraZeros] 
        sent.extend(extraZeros)
    elif len(sent) > desLen:    #truncate
        sent = sent[:desLen]
    
    #>>> Get embeddings for each word | word = row , col = emb values
    sent_emb = [] #contains all a word embeddings for each word in 'sent' that we will return
    for word in sent:
        if emb.get(word) is None: 
          sent_emb.append(emb.get(str(0)))
        else:
          sent_emb.append(emb.get(word))
    
    return sent_emb

def read_embeddings(n=1000):
    #Fuentes: Reads n embeddings from file
    #Fuentes: Returns a dictionary were embedding[w] is the embeding of string w
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
    
    allSentences = [] #contains all the sentences or word lists from all 3 books combined
    numSentEachBook = [] # number of sentences in each book | index 0 = num sentences in book 0
    for u, url in enumerate(url_list):
        word_lists = get_sentence_list(url)
        print('Book {} contains {} sentences'.format(u,len(word_lists)))
        lengths = np.array([len(wl) for wl in word_lists])
        print('Sentence length stats (min,max and mean words in a sentence):')
        print('min = {} max = {} mean = {:4f}'.format(np.min(lengths),np.max(lengths),np.mean(lengths)))
        allSentences.extend(word_lists)
        numSentEachBook.append(len(lengths)) #len(lengths) = total number of sentences for book 'u'
        
    print('Total number of sentences in all 3 books: ', len(allSentences))
    
    vocabulary_size = 22500        
    embedding = read_embeddings(vocabulary_size)
    
    #Fuentes: See if the protagonists appear in the embedding list    
    #Fuentes: I recommend increasing vocabulary size until all 3 appear in vocabulary
    for w in ['buck','dracula','holmes']:
        try:
            print(w,'embedding:\n',embedding[w])
        except:
            print(w,'is not in dictionary')
            pass
    
    #Each sentence is mapped to a book > 2D array mapped to a number(1-3)
    
    #get each word embedding from each book - contains duplicated embeddings
    desiredLength = 7
    all_word_emb = [] #should be a list of lists
    for sent in allSentences:
        all_word_emb.append(sent_embedder(sent, embedding,desiredLength))

    print("Total sentence embeddings calculated: ",len(all_word_emb)) #should be 12260 bc thats how many sentences are in all 3 books combined
    
    
    #Seperate data into X and y
    
    X = np.asarray(all_word_emb)
    y = [] # create a one hot rep of the data | [1,0,0] means book 0, [0,1,0] means book 1, [0,0,1] means book 2
    
    #numSentEachBook is a list where index 0(book zero) contains number of sentences for that book and so on
    for book in range(len(numSentEachBook)):
      for sent in range(numSentEachBook[book]):
        ohRep = np.zeros(3) #3 bc there are 3 classes | book 0,1,2
        ohRep[book] = 1
        y.append(ohRep)
     
    #Convert to np array so keras conv1D can understand it
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)
    
    #---------------- REST OF CODE IN THE 'NLP_RNoahPadilla.ipynb' because CNN
    