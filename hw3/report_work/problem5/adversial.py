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

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    for i in range(num_step):
        loss_value, grads_value = iter_func([input_image_data,False])
        input_image_data += grads_value * 1
    return input_image_data

def main():
    emotion_classifier = load_model('../../cnn_model')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ["conv2d_4"]
    collect_layers = [ layer_dict[name].output for name in name_ls ]
    nb_filter = 48
    num_step=200
    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(nb_filter)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            #iterate = K.function([input_img], [target, grads])
            iterate = K.function([input_img,K.learning_phase()], [target, grads])
            
            filter_imgs[filter_idx].append(grad_ascent(num_step, input_img_data, iterate))

        fig = plt.figure(figsize=(14, 8))
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(filter_imgs[i][0].reshape(48,48), cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            #plt.xlabel('{:.3f}'.format(filter_imgs[i][1]))
            plt.tight_layout()
        fig.suptitle('Filters of layer {} (# Ascent Epoch 200)'.format(name_ls[cnt]))
        fig.savefig('e87')

if __name__ == "__main__":
    main()
