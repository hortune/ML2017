import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, GRU
from keras.models import Sequential, load_model
from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

TEST_FILE = sys.argv[1]

import pickle
f = open('2_data','rb')
tokenizer, tags, all_tags, tag2int, int2tag, x_trainshape = pickle.load(f)
f.close()

y_train_data = np.zeros((len(tags), len(all_tags)))
for i in range(len(tags)):
    int_tags = [tag2int[t] for t in tags[i]]
    y_train_data[i, int_tags] = 1

model = load_model('2_model')

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

new_x_test = np.zeros((x_test.shape[0], x_trainshape[1]))
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

f = open(sys.argv[2], 'w')
print('"id","tags"', file=f)
for i, ts in enumerate(y_tags):
    print('"{}","'.format(i), end='', file=f)
    for i, t in enumerate(ts):
        if i != 0:
            print(' ', end='', file=f)
        print(t, end='', file=f)
    print('"', file=f)
f.close()
