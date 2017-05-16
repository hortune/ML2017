from gensim import corpora
import tensorflow as tf
import sys
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
import numpy as np

from keras.callbacks import Callback
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical
import keras.preprocessing.image as img
#categorical_crossentropy
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
def f1score(y_true,y_pred):
    print y_pred
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1
def load_data():
        k = open('train_data.csv','r').readlines()
        x = []
        labels_index = {}
        labels = []
        for string in k[1:]:
            pre = string.find('"')
            last = string.find('"',pre+1)
            y_text  = string[pre+1:last].split()
            label = []
            for text in y_text:
                if text not in labels_index:
                    lid = len(labels_index)
                    labels_index[text] = lid
                label.append(labels_index[text])
            labels.append(label)

            x.append(string[last+2:])

        y = open('test_data.csv','r').readlines()
        x_test = [] 
        for string in y[1:]:
            pre = string.find(',')
            x_test.append(string[pre+1:])

        #tokenizer = Tokenizer(num_words=50000)
        #tokenizer.fit_on_texts(x)
        #sequences = tokenizer.texts_to_sequences(x)
        #x_test = tokenizer.texts_to_sequences(x_test)
        #x_test = pad_sequences(x_test,padding='post',maxlen=310)

        #word_index = tokenizer.word_index
        #data = pad_sequences(sequences,padding='post', maxlen=310)
        #
        stoplist = set('for a of the and to in'.split())
        raw_corpus = x
        texts = text_to_word_sequence(x,lower=True,split=" ") 
        texts = [[word for word in document if word not in stoplist] for document in texts]
        
        #
        
        new_labels = []
        for i in labels:
            arr = np.zeros(len(labels_index))
            arr[i]=1
            new_labels.append(arr)
        labels = np.array(new_labels)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        return data,labels

x,labels = load_data()

m2 = Sequential()
m2.add(Embedding(50000,100,input_length=310))

m2.add(LSTM(70,return_sequences=True))
m2.add(LSTM(50,dropout=0.4))
m2.add(Dense(40,activation='relu'))
m2.add(Dropout(0.4))
m2.add(Dense(38,activation='sigmoid'))

#sgd = SGD(lr=0.1,momentum=0.5,clipvalue=0.5)
m2.summary()
m2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['categorical_accuracy',f1score])

#
nb_validation_samples = int(0.2*x.shape[0])
x_train = x[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = x[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
#
m2.fit(x_train,y_train,epochs=200,batch_size=128,validation_data=(x_val,y_val))
res = m2.predict(x_val)

