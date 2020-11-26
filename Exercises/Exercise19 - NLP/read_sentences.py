'''
Author: Olac Fuentes
Modified by: R Noah Padilla

The goal of this assignment is written below.

The program read_sentences.py reads sentences form online classic books and 
converts them to a list of sentences, where each sentence is a list of words.
    


    1. Write a function that receives a sentence and returns a 2D array containing 
        the embeddings of the words in the sentence. Your function should receive the embeddings 
        dictionary, the sentence and the desired length of the representation; if the 
        sentence is shorter than the desired length, path the array with zeros; if it’s longer, 
        truncate the representation.
    
    2. Apply the function to produce an embedding representation of each of the 
        sentences in the three books used in the read_sentences.py program and generate
        a dataset containing examples of 3 classes, one for each book.
    
    3. Randomly split the data into training and testing.
    
    4. Train and test a system to determine the book each sentence belongs to.
    
    CLASSIFICATION PROBLEM

'''


import bs4 as bs
import urllib.request
import numpy as np

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
    for u, url in enumerate(url_list):
        word_lists = get_sentence_list(url)
        print('Book {} contains {} sentences'.format(u,len(word_lists)))
        lengths = np.array([len(wl) for wl in word_lists])
        print('Sentence length stats:')
        print('min = {} max = {} mean = {:4f}'.format(np.min(lengths),np.max(lengths),np.mean(lengths)))
        
   
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
            