'''
Exam 2 - only create code for word_analogy function - finished b4 4:30
'''
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

#perform equation then find euclidean distance
def word_analogy(pair,w,options,emb):
    
    #E[newcountry] + E[capital] - E[country]
    wa = emb[w]+emb[pair[1]]-emb[pair[0]]
    
    #Set the first value as a default similar thing - will update if its not the most similar
    mostSimilarThing = options[0]
    mostSimilarThingEmb = emb[mostSimilarThing]
    smallestDiff = np.linalg.norm(wa - mostSimilarThingEmb)
    for thing in options:
        dist = np.linalg.norm(wa - emb[thing])
        if (dist < smallestDiff) and (thing != w):
            mostSimilarThing = thing
            mostSimilarThingEmb = emb[mostSimilarThing]
            smallestDiff = dist
    S = '{} is to {} as {} is to {}'.format(pair[0],pair[1],w,mostSimilarThing)
    return S
    
if __name__ == "__main__":  
    vocabulary_size = 30000        
    embedding = read_embeddings(vocabulary_size)
    
    pair = ['france','paris']
    w = 'spain'
    options = ['washington','berlin','moscow','madrid']
    print(word_analogy(pair,w,options,embedding))
    
    pair = ['woman','queen']
    w = 'man'
    options = ['president','lord','minister','politician','king']
    print(word_analogy(pair,w,options,embedding))
    
    pair = ['princess','queen']
    w = 'prince'
    options = ['president','lord','minister','politician','king']
    print(word_analogy(pair,w,options,embedding))
    
    pair = ['algorithm','program']
    w = 'recipe'
    options = ['food','restaurant','taco','banana']
    print(word_analogy(pair,w,options,embedding))
    
    pair = ['cat','dog']
    w = 'tiger'
    options = ['lion','wolf','coyote','whale','dolphin']
    print(word_analogy(pair,w,options,embedding))
    
    pair = ['france','french']
    w = 'germany'
    options = ['spanish','english','russian','persian','german']
    print(word_analogy(pair,w,options,embedding))

    pair = ['child','person']
    w = 'colt'
    options = ['lion','wolf','coyote','whale','horse']
    print(word_analogy(pair,w,options,embedding))

    pair = ['old','new']
    w = 'fast'
    options = ['slow','planet','berlin','earth','king','german']
    print(word_analogy(pair,w,options,embedding))

    pair = ['toe','foot']
    w = 'finger'
    options = ['slow','planet','hand','earth','king','german']
    print(word_analogy(pair,w,options,embedding))

    pair = ['positive','negative']
    w = 'proton'
    options = ['neutron','electron','atom','molecule']
    print(word_analogy(pair,w,options,embedding))

'''
france is to paris as spain is to madrid
woman is to queen as man is to king
princess is to queen as prince is to king
algorithm is to program as recipe is to food
cat is to dog as tiger is to wolf
france is to french as germany is to german
child is to person as colt is to horse
old is to new as fast is to slow
toe is to foot as finger is to hand
positive is to negative as proton is to electron
'''