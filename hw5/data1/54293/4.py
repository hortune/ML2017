from __future__ import print_function
import numpy as np
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.recurrent import LSTM,GRU
from keras.losses import binary_crossentropy
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
import os
import keras.backend as K
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
from nltk.corpus import stopwords
import re
def load_data():
    label_sets=[]
    text_sets=[]
    test_sets=[]
    dic = {}
    label_list = []
    test_b_train = {}
    train_b_test = {}
    for string in open('test_data.csv','r').readlines()[1:]:
        num,words=string.split(',',1)
        
        words = " ".join([word for word in words.split() if "http" not in word])
        words = re.sub("[^a-zA-Z]"," ",words).lower().split()
        words = [w for w in words if w not in stopwords.words("english")]
        for i in words:
            test_b_train[i] = True
        
        test_sets.append( words )             
        
    for string in open('train_data.csv','r').readlines()[1:]:
        num,label,words= string.split(',',2)
        
        words = " ".join([word for word in words.split() if "http" not in word])
        words = re.sub("[^a-zA-Z]"," ",words).lower().split()
        for i in words:
            train_b_test[i] = True
        words = [w for w in words if w not in stopwords.words("english") and w in test_b_train]

        labels = label[1:-1].split()
        new_label = []
        for label in labels:
            if label not in dic:
                dic[label]=len(dic)
                label_list.append(label)
            new_label.append(dic[label])
        lab = np.zeros(38)
        lab[new_label] = 1
        label_sets.append(lab)
        
        text_sets.append( " ".join(words) )
    new_test_sets = []
    for string in test_sets:
        words = [word for word in string if word in train_b_test]
        new_test_sets.append(" ".join(words) )            
    return text_sets,np.array(label_sets),new_test_sets,dic,label_list

def get_embedding_dict():
    embedding_dict = {}
    with open('/tmp2/hortune/glove.6B.100d.txt',"r") as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

X_data,Y_data,X_test,dic_for_label,dic2 = load_data()
all_corpus = X_data + X_test
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_corpus)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)

train_sequences = pad_sequences(train_sequences)
max_article_length = train_sequences.shape[1]
print (max_article_length)
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

num_words = len(tokenizer.word_counts)
embedding_dict = get_embedding_dict()

embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,100)


def f1score(y_true, y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true*y_pred)
    if tp == 0:
        return 0
    
    precision = tp/(K.sum(y_pred))
    recall = tp/(K.sum(y_true))
    return 2.0*((precision*recall)/(precision+recall))

model = Sequential()
model.add( Embedding(num_words,100,weights=[embedding_matrix],input_length=max_article_length,trainable=False) )
model.add(GRU(128,dropout=0.1,activation='tanh'))
model.add( Dense(output_dim = 256, activation='relu') )
model.add( Dropout(0.1))
model.add( Dense(output_dim = 128, activation='relu') )
model.add( Dropout(0.1))
model.add( Dense(output_dim = 64, activation='relu') )
model.add( Dropout(0.1))
model.add( Dense(output_dim = 38) )
model.add( Activation('sigmoid') )

model.compile( loss='categorical_crossentropy', optimizer=Adam(lr=0.001,decay=1e-6,clipvalue=0.5), metrics=[f1score] )
model.fit(train_sequences, Y_data, batch_size=128, epochs = 28)
#model.save("model.h5")

ans = model.predict(test_sequences)
out = np.zeros(ans.shape)
out[ans>0.4] = 1

count = 0
f = open("out1.csv", "w")
print('"id","tags"', file = f)
for i in range(1234):
    print('"', i, '"', sep='', end = ',', file = f)
    tmp = '"'
    for w in range(38):
        if out[i, w] == 1:
            tmp += dic2[w]
            tmp += ' '
    if len(tmp) !=1: 
        print(tmp[:len(tmp)-1], end = '"\n', file = f)
    else:
        print("\"\"",file=f)
f.close()
