from keras import applications
from keras import backend as K
import numpy as np
from scipy.misc import imsave

from keras.utils import plot_model
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
from matplotlib import pyplot as plt

# build the VGG16 network
model = load_model('../problem4/lastcnn_model')
input_img = model.input #x_train.reshape(x_train.shape[0],1,48,48,1)

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'conv2d_2'
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
def deprocess_image(x):
            # normalize tensor: center on 0., ensure std is 0.1
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 0.1

            # clip to [0, 1]
            x += 0.5
            x = np.clip(x, 0, 1)

            # convert to RGB array
            x *= 255

            x = x.transpose((1, 2, 0))
            x = np.clip(x, 0, 255).astype('uint8')
            return x




for filter_index in range(1):
    #filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img,K.learning_phase()], [loss, grads])
        # we start from a gray image with some noise
        input_img_data = (np.random.random((1, 48, 48,1))-0.5)*20+128
        # run gradient ascent for 20 steps
        for i in range(100):
            loss_value, grads_value = iterate([input_img_data,False])
            input_img_data += grads_value * 1

        # util function to convert a tensor into a valid image

        img = input_img_data[0]
        img = deprocess_image(img).reshape(48,48)
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img,cmap='BuGn')

        fig =plt.figure()
        plt.imshow(img,cmap='BuGn')
        #fig = plt.gcf()
        #plt.draw()
        fig.savefig('privateTest{}.png'.format(filter_index), dpi=100)
        plt.close(fig)
