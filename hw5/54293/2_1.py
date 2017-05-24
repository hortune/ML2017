import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, GRU
from keras.models import Sequential, load_model
from keras.models import Model


TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
OUTPUT = '2'

EMBEDDING_DIM = 100
GLOVE = 'd:/ML_data/HW5/data/glove.6B.100d.txt'

glove = {}
f = open(GLOVE, 'r', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove[word] = coefs
f.close()

all_tags = []
tags = []
txts = []

f = open(TRAIN_FILE, 'r', encoding='utf8')
lines = f.readlines()[1:]
f.close()
for i in range(len(lines)):
    lines[i] = lines[i].split('"')[1:]
    for rest in lines[i][2:]:
        lines[i][1] += rest

    line_tag = lines[i][0].split(' ')
    all_tags += line_tag
    tags += [line_tag]
    txts += [ lines[i][1] ]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(txts)
sequences = tokenizer.texts_to_sequences(txts)

word_index = tokenizer.word_index

x_train_data = pad_sequences(sequences)
all_tags = set(all_tags)
tag2int = {tag:i for i, tag in enumerate(all_tags)}
int2tag = {i:tag for i, tag in enumerate(all_tags)}

y_train_data = np.zeros((len(tags), len(all_tags)))
for i in range(len(tags)):
    int_tags = [tag2int[t] for t in tags[i]]
    y_train_data[i, int_tags] = 1

select_vali = -500
x_train = x_train_data[:select_vali]
y_train = y_train_data[:select_vali]
x_vali  = x_train_data[select_vali:]
y_vali  = y_train_data[select_vali:]

num_words = len(word_index)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= len(word_index):
        continue
    embedding_vector = glove.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
def generate_model():
    model = Sequential()

    model.add(Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=x_train.shape[1], trainable=False))
    model.add(GRU(128, activation='elu', dropout=0.25))
    model.add(Dense(32, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(len(all_tags), activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
model = generate_model()

model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_vali, y_vali))


test_txts = []

f = open(TEST_FILE, 'r', encoding='utf8')
lines = f.readlines()[1:]
f.close()
for i in range(len(lines)):
    lines[i] = lines[i].split(',')[1:]
    for rest in lines[i][1:]:
        lines[i][0] += rest
    test_txts += [ lines[i][0] ]

test_sequences = tokenizer.texts_to_sequences(test_txts)
x_test = pad_sequences(test_sequences)


new_x_test = np.zeros((x_test.shape[0], x_train.shape[1]))
new_x_test[:,:x_test.shape[1]] = x_test

y_prob = model.predict(new_x_test)

y_prob_max = np.max(y_prob, axis=1)
y_thres = y_prob_max * 0.25
y_tags = []
for i in range(y_prob.shape[0]):
    tag = []
    for j in range(38):
        if y_prob[i][j] > y_thres[i]:
            tag += [int2tag[j]]
    y_tags += [tag]

f = open('submission.csv', 'w')
print('"id","tags"', file=f)
for i, ts in enumerate(y_tags):
    print('"{}","'.format(i), end='', file=f)
    for i, t in enumerate(ts):
        if i != 0:
            print(' ', end='', file=f)
        print(t, end='', file=f)
    print('"', file=f)
f.close()
