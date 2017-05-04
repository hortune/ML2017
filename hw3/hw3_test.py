import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))
import numpy as np

from keras.callbacks import Callback
from numpy import genfromtxt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import keras.preprocessing.image as img
#categorical_crossentropy
def load_raw_data(name):
        file_name = open(name,'r')
        x = []
        y = []
        for line in file_name.readlines()[1:]:
            y.append(int(line.split(',')[0]))
            x.append(list(map(int,line.split(',')[1].split())))
        return np.array(x),np.array(y)

def load_data():
        x_test,y_test = load_raw_data(sys.argv[1])
        x_test = x_test.astype('float32')
        x_test = x_test/255
        return x_test

x_test=load_data()
x_test = x_test.reshape(x_test.shape[0],48,48,1)


model2=load_model('cnn_model')
file_name = open(sys.argv[2],'w')
res = model2.predict_classes(x_test)

file_name.write("id,label\n")

for i in range(len(res)):
    file_name.write(str(i)+','+str(res[i])+'\n')
