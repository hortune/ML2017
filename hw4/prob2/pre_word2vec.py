import word2vec
#word2vec.word2phrase('all.txt','allphrase.txt',verbose=True)
word2vec.word2vec('all.txt','model.bin',50,min_count=50,verbose=False)
