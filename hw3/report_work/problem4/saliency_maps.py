import os
from termcolor import colored, cprint
import argparse
from keras.utils import plot_model
from keras.models import load_model
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
import numpy as np
from matplotlib import pyplot as plt


from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from vis.visualization import visualize_saliency

# Build the VGG16 network with ImageNet weights
model = load_model('../problem3/lastcnn_model')#VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
#layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
for idx, layer in enumerate(model.layers):
    print idx, layer.name
# Images corresponding to tiger, penguin, dumbbell, speedboat, spider

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

x_train,y_train=load_data()
x_train = x_train.reshape(x_train.shape[0],48,48,1)

#x_train = x_train/255.
heatmaps = []
index_pool=[15000]
for i in index_pool:
        seed_img = np.copy(x_train[i])
        x = x_train[i]/255
        pred_class = np.argmax(model.predict(x_train)[i])
        heatmap = visualize_saliency(model, 9, [pred_class],seed_img,1)
        heatmaps.append(heatmap)
        heatmap = visualize_saliency(model, 9, [pred_class],seed_img,0)
        heatmaps.append(heatmap)
        heatmap = visualize_saliency(model, 9, [pred_class],seed_img,0.5)
        heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()
"""
plt.figure()
plt.imshow(heatmap, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
plt.show()
"""
