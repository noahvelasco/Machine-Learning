import bs4 as bs
import urllib.request
import numpy as np
import math

def read_embeddings(n=1000):
    # Reads n embeddings from file
    # Returns a dictionary were embedding[w] is the embeding of string w
    embedding = {}
    count = 0
    with open('glove.6B.50d.txt', encoding="utf8") as f: 
        for line in f: 
            ls = line.split(" ")
            emb = [np.float32(x) for x in ls[1:]]
            embedding[ls[0]]=np.array(emb) 
            count+=1    
            if count>= n:
                break
    return embedding

def most_similar(w,emb):
    
    try:
        #Set the first value as a default similar word - will update if its not the most similar
        mostSimilarWord = list(emb.keys())[0]
        mostSimilarEmb = emb[mostSimilarWord]
        
        smallestDiff = np.linalg.norm(emb[w] - mostSimilarEmb)
        
        for word in emb:
            dist = np.linalg.norm(emb[w] - emb[word])
            if (dist < smallestDiff) and (word != w):
                mostSimilarWord = word
                mostSimilarEmb = emb[word]
                smallestDiff = dist
        return mostSimilarWord
    except:
        return '---'

    
if __name__ == "__main__":  
    vocabulary_size = 30000
    embedding = read_embeddings(vocabulary_size)
    
    for word in ['white', 'coyote', 'spain', 'football','taco','university','convolutional']:
        ms =  most_similar(word,embedding)
        print('The most similar word to {} is {}'.format(word,ms))
        
        
'''
The most similar word to white is black
The most similar word to coyote is wolf
The most similar word to spain is portugal
The most similar word to football is soccer
The most similar word to taco is burger
The most similar word to university is college
The most similar word to convolutional is ---
'''