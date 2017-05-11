import numpy as np
import word2vec
import nltk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = word2vec.load('model.bin')
words, vectors = model.vocab,model.vectors

pca = PCA(n_components=2)
pca.fit(vectors)
# Gen the trans_vec
trans_vec = pca.transform(vectors)

# add tag 
text = nltk.pos_tag(words)
k_vec = []
prohibit=set("|”“()‘`,.:;'!?\"")
allow = set(["JJ","NNP","NN","NNS"])
#plt.scatter(trans_vec[:,0],trans_vec[:,1])
for word,vec in zip(text[:500],trans_vec[:500]):
    #print( word[0],word[1])
    if set(word[0])&prohibit or word[1] not in allow:
        continue
    print( word[0],word[1])
    plt.annotate(word[0],xy=vec,xytext=(0,0),textcoords='offset points')
    k_vec.append(list(vec))

plt.scatter(np.array(k_vec)[:,0],np.array(k_vec)[:,1])
plt.show()
