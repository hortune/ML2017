import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
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
from keras.models import load_model

import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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
        x_train = x_train/255
        x_test = x_test/255
        return x_train[22000:],y_train[22000:]

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    emotion_classifier = load_model('lastcnn_model')
    np.set_printoptions(precision=2)
    x_validate,y_validate=load_data()
    x_validate = x_validate.reshape(x_validate.shape[0],48,48,1)
    dev_feats = x_validate#= read_dataset('valid')
    predictions = emotion_classifier.predict_classes(dev_feats)
    te_labels = y_validate#get_labels('valid')
    print "predictions",predictions
    conf_mat = confusion_matrix(te_labels,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.savefig('second_kind.png')
    plt.show()
main()
