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
from keras.utils import np_utils
import keras.preprocessing.image as img
#categorical_crossentropy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


k = open('train_data.csv','r').readlines()
x = []
y = []
for string in k[1:]:
    pre = string.find('"')
    last = string.find('"',pre+1)
    y.append(string[pre+1:last])
    x.append(string[last+2:])

tokenizer = Tokenizer(nb_words=100)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

"""
x_train = x_train.reshape(x_train.shape[0],48,48,1)
model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

datagen = img.ImageDataGenerator(
	rotation_range = 3,
	horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)
datagen.fit(x_train)
model2.fit_generator(datagen.flow(x_train,y_train,batch_size=128),steps_per_epoch=len(x_train)/16,epochs=120)

model2.save('cnn_model')
"""
