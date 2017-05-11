import word2vec
#word2vec.word2phrase('all.txt','allphrase.txt',verbose=True)
word2vec.word2vec('all.txt','model.bin',100,verbose=True)
