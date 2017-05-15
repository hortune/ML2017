import tensorflow as tf
import sys
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
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

        tokenizer = Tokenizer(num_words=50000)
        tokenizer.fit_on_texts(x)
        sequences = tokenizer.texts_to_sequences(x)

        word_index = tokenizer.word_index
        max_num = 0
        data = pad_sequences(sequences, maxlen=175)

        # For shuffle
        #labels = to_categorical(np.array(labels))
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


        """
        nb_validation_samples = int(0.2*data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]
        """
        return data,labels

x,labels = load_data()

model = Sequential()
model.add(Embedding(50000,50,input_length=175))
model.compile('rmsprop','mse'output_array = model.predict(x)

data = model.predict(x)

nb_validation_samples = int(0.2*data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# Done Preprocessing
# I may need to use word2vec


