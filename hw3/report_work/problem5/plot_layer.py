#!/usr/bin/env python
# -- coding: utf-8 --
from scipy.misc import imsave
from keras.utils import plot_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
import numpy as np

def load_raw_data(name):
        file_name = open(name,'r')
        x = []
        y = []
        for line in file_name.readlines()[1:]:
            y.append(int(line.split(',')[0]))
            x.append(list(map(int,line.split(',')[1].split())))
        return np.array(x),np.array(y)

def load_data():
        x_train, y_train = load_raw_data('/tmp/train.csv')
        x_train = x_train.astype('float32')

        x_train = x_train/255
        return x_train

def main():
    x_train=load_data()
    emotion_classifier = load_model('../../cnn_model')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    
    input_img = emotion_classifier.input
    name_ls = ["conv2d_4"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    private_pixels = x_train.reshape(x_train.shape[0],1,48,48,1)

    choose_id = 17
    photo = private_pixels[choose_id]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(48):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        fig.savefig('layer{}'.format(cnt))
main()
