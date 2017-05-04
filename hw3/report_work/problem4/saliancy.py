#!/usr/bin/env python
# -- coding: utf-8 --

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
"""
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')
"""

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
        x_test,y_test = load_raw_data('/tmp/test.csv')
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        #y_train = np_utils.to_categorical(y_train, 7)
        #x_train = x_train/255
        #x_test = x_test/255
        return x_train,y_train

#x_train,y_train=load_data()
#x_train = x_train.reshape(x_train.shape[0],48,48,1)
def main():
        
    x_train,y_train=load_data()
    private_pixels = x_train.reshape(x_train.shape[0],1,48,48,1)
    
    emotion_classifier = load_model('lastcnn_model')
    """
    private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
                       for i in range(len(private_pixels)) ]
    """
    input_img = emotion_classifier.input
    img_ids = [1]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = None
        
        print "fn", fn 
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''

        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('TprivateTest{}.png'.format(idx), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('privateTest{}.png'.format(idx), dpi=100)

if __name__ == "__main__":
    main()
