import numpy as np
import word2vec
import nltk
from sklearn.decomposition import PCA
model = word2vec.load('model.bin')
words, vectors = model.vocab,model.vectors

pca = PCA(n_components=2)
pca.fit(vectors)
# Gen the trans_vec
trans_vec = pca.transform(vectors)

# add tag 
text = nltk.pos_tag(words)

prohibit=set("|”“()‘`,.:;'!?\"")
allow = set(["JJ","NNP","NN","NNS"])

for word,vec in zip(text,trans_vec):
    #print( word[0],word[1])
    if set(word[0])&prohibit or word[1] not in allow:
        continue
    print( word[0],word[1])
