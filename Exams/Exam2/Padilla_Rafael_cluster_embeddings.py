'''
Exam 2 - create data X and fit it into Kmeans model and print out the closest thing to each cluster
'''
from sklearn.cluster import KMeans
import numpy as np
import math

def read_embeddings(n=1000):
    # Reads n embeddings from file
    # Returns a dictionary where embedding[w] is the embeding of string w
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
    
if __name__ == "__main__":
    
    vocabulary_size = 30000
    embedding = read_embeddings(vocabulary_size)
    
    # Create array X of size (vocabulary_size,50) containing embeddings
    X = np.array([embedding[emb] for emb in embedding])

    # Cluster the data
    model = KMeans(n_clusters=10, random_state=0).fit(X)
    
    # Retrieve cluster centers, an array of size  (10,50)
    centers = model.cluster_centers_
    
    # For each c in centers, find the word whose embedding is most similar to c and print it
    for i,c in enumerate(centers):
        #Set the first value as a default similar thing - will update if its not the most similar
        mostSimilarThing = list(embedding.keys())[0]
        mostSimilarThingEmb = embedding[mostSimilarThing]
        smallestDiff = np.linalg.norm(c - mostSimilarThingEmb)
        
        for word in embedding:
            dist = np.linalg.norm(c - embedding[word])
            if (dist < smallestDiff):
                mostSimilarWord = word
                mostSimilarEmb = embedding[word]
                smallestDiff = dist
        print("> Closest thing to center", i , "is", mostSimilarWord )
'''
> Closest thing to center 0 is importantly
> Closest thing to center 1 is teaming
> Closest thing to center 2 is incidentally
> Closest thing to center 3 is bates
> Closest thing to center 4 is lastly
> Closest thing to center 5 is moreover
> Closest thing to center 6 is alluding
> Closest thing to center 7 is 525
> Closest thing to center 8 is ostensibly
> Closest thing to center 9 is aforementioned
'''