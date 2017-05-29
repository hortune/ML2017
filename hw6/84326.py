#coding = utf8
# 1.41145
import sys
import pandas as pd
import numpy as np
def load_data(tpath):
    test_data = pd.read_csv(tpath)
    test_data.UserID = test_data.UserID.astype('category')
    test_data.MovieID = test_data.MovieID.astype('category')

    return test_data

import keras.models as kmodel
import keras.backend as K
import keras
import tensorflow as tf
from keras import regularizers
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
def generate_model(n_movies, n_users):
        movie_input = keras.layers.Input(shape=(1,))
        #movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 120,embeddings_regularizer=regularizers.l2(1e-2))(movie_input))
        movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 120, embeddings_initializer='random_uniform')(movie_input))
        movie_vec = keras.layers.Dropout(0.2)(movie_vec)
        user_input = keras.layers.Input(shape=(1,))
        #user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 120,embeddings_regularizer=regularizers.l2(1e-2))(user_input))
        user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 120, embeddings_initializer='random_uniform')(user_input))
        user_vec = keras.layers.Dropout(0.2)(user_vec)
        input_vecs = keras.layers.merge([movie_vec,user_vec],'dot') 
        #input_vecs = keras.layers.Dropout(0.2)(input_vecs)
        #input_vecs = keras.layers.Dense(5, activation='softmax')(input_vecs)
        model = kmodel.Model([movie_input, user_input], input_vecs)
        #model.compile(optimizer = 'adam',loss = 'categorical_crossentropy')
        model.compile(optimizer = 'adam',loss = 'mean_squared_error')
        #model.summary()
        return model
import os
if __name__ == '__main__':
    test = load_data(os.path.join(sys.argv[1],'test.csv'))
    
    m_test = test.MovieID
    u_test = test.UserID
    
    model = kmodel.load_model('84326.h5py')#generate_model(3952, n_users)
    y_pred = np.clip(model.predict([m_test,u_test]),1,5)
    output = pd.DataFrame({'TestDataID':np.arange(len(y_pred)+1)[1:],'Rating':y_pred.flatten()}, columns = ['TestDataID','Rating'])   
    output.to_csv(sys.argv[2],index=False)
