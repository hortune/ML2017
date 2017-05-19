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

def load_data():
    label_sets=[]
    text_sets=[]
    test_sets=[]
    for string in open('test_data.csv','r').readlines()[1:]:
        num,words=string.split(',',1)
        test_sets.append( words )            
    
    for string in open('train_data.csv','r').readlines()[1:]:
        num,label,words= string.split(',',2)
        label_sets.append(label[1:-1].split())
        text_sets.append( words )
    return text_sets,label_sets,test_sets

def get_embedding_dict():
    embedding_dict = {}
    with open('glove.6B.100d.txt',"r") as f:
        for line in f:
            values = line.split("")
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

X_data,Y_data,X_test = load_data()
all_corpus = X_data + X_test
tokenizer_word = Tokenizer()
tokenizer_word.fit_on_texts(all_corpus)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)

train_sequences = pad_sequences(train_sequences)
max_article_length = train_sequences.shape[1]
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

embedding_dict = get_embedding_dict()
embedding_matrix = get_embedding_matrix(word_index,embedding_dict,50000,100)



"""
def f1score(y_true, y_pred):
    thresh = 0.3
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true*y_pred)
    if tp == 0:
        return 0
    precision = tp/K.sum(y_pred)
    recall = tp/K.sum(y_true)
    return 2.0*(precision*recall)/(precision+recall)

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAXLEN, trainable=False)


model = Sequential()

model.add( embedding_layer )
model.add(GRU(128,dropout=0.1,activation='tanh'))
model.add( Dense(output_dim = 256, activation='elu') )
model.add( Dropout(0.1))
model.add( Dense(output_dim = 128, activation='elu') )
model.add( Dropout(0.1))
model.add( Dense(output_dim = 64, activation='elu') )
model.add( Dropout(0.1))
model.add( Dense(output_dim = 38) )
model.add( Activation('sigmoid') )

model.compile( loss='categorical_crossentropy', optimizer=Adam(lr=0.001,decay=1e-6,clipvalue=0.5), metrics=[f1score] )

#earlystopping = EarlyStopping(monitor='val_f1_score',patience = 10, verbose = 1, mode='max')
#checkpoint = ModelCheckpoint(filepath='')


model.fit(x_train, y_train, batch_size=128, epochs = 130, validation_split=0.1)

#model.save("model.h5")
ans = model.predict(x_test)
out = np.zeros(ans.shape)
out[ans>0.3] = 1

count = 0
f = open("out.csv", "w")
print('"id","tags"', file = f)
for i in range(1234):
    print('"', i, '"', sep='', end = ',', file = f)
    tmp = '"'
    for w in range(38):
        if out[i, w] == 1:
            tmp += all_label[w]
            tmp += ' '
    print(tmp[:len(tmp)-1], end = '"\n', file = f)

f.close()
"""
