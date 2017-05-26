#coding = utf8
import pandas as pd
import numpy as np
from IPython import embed
def load_data(upath, mpath, rpath, tpath):
    users = pd.read_csv(upath).sort_values('UserID')
    movies = []
    with open(mpath,'r',encoding="latin1") as fd:
        for line in fd:
            word = line.replace('\n','')
            word = word.split(',')
            movies.append({'movieid':word[0],'title':','.join(word[1:-1]),'genre':word[-1]})
    movies = pd.DataFrame(movies)        
    movies['genre'] = movies.genre.str.split('|')

    ratings = pd.read_csv(rpath)

    users.Age = users.Age.astype('category')
    users.Gender = users.Gender.astype('category')
    users.Ocuppation = users.Occupation.astype('category')
    
    ratings.UserID = ratings.UserID.astype('category')
    ratings.MovieID = ratings.MovieID.astype('category')
    test_data = pd.read_csv(tpath)
    test_data.UserID = test_data.UserID.astype('category')
    test_data.MovieID = test_data.MovieID.astype('category')

    return users, movies, ratings, test_data

import keras.models as kmodel
import keras.backend as K
import keras
from sklearn import cross_validation
import tensorflow as tf
from keras import regularizers
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def generate_model(n_movies, n_users):
        movie_input = keras.layers.Input(shape=(1,))
        #movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 120,embeddings_regularizer=regularizers.l2(1e-2))(movie_input))
        movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 100)(movie_input))
        movie_vec = keras.layers.Dropout(0.2)(movie_vec)
        user_input = keras.layers.Input(shape=(1,))
        #user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 120,embeddings_regularizer=regularizers.l2(1e-2))(user_input))
        user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 100)(user_input))
        user_vec = keras.layers.Dropout(0.2)(user_vec)
        input_vecs = keras.layers.merge([movie_vec,user_vec],'dot') 
        input_vecs = keras.layers.Dropout(0.2)(input_vecs)
        #input_vecs = keras.layers.Dense(5, activation='softmax')(input_vecs)
        model = kmodel.Model([movie_input, user_input], input_vecs)
        #model.compile(optimizer = 'adam',loss = 'categorical_crossentropy')
        model.compile(optimizer = 'adam',loss = 'mean_squared_error',metric=[root_mean_squared_error])
        #model.summary()
        return model
if __name__ == '__main__':
    users, movies, ratings, test = load_data('users.csv','movies.csv','train.csv','test.csv')
    n_movies = movies.shape[0]
    n_users = users.shape[0]
    
    movieid = ratings.MovieID.cat.codes.values
    userid = ratings.UserID.cat.codes.values
    #y = np.zeros((ratings.shape[0],5))
    y = ratings.Rating
    #y[np.arange(ratings.shape[0]), ratings.Rating -1] = 1
    
    m_test = test.MovieID.cat.codes.values
    u_test = test.UserID.cat.codes.values
    
    model = generate_model(4000, n_users)

    m_train,m_val,u_train,u_val,y_train,y_val = cross_validation.train_test_split(movieid,userid,y,test_size=0.5)
   
    model.fit([m_train,u_train],y_train,epochs=10,verbose=2,batch_size=20000,validation_data = ([m_val,u_val],y_val))
    #y_pred =np.argmax( model.predict([movieid,userid]),1)+1
    y_pred = model.predict([movieid,userid])
    #y_pred = model.predict([m_test,u_test])
    output = pd.DataFrame({'TestDataID':np.arange(len(y_pred)+1)[1:],'Rating':y_pred.flatten()}, columns = ['TestDataID','Rating'])   
    output.to_csv('result.csv',index=False)
    output = pd.DataFrame({'TestDataID':np.arange(len(y_pred)+1)[1:],'Rating':np.round(y_pred.flatten())}, columns = ['TestDataID','Rating'])   
    output.to_csv('result_round.csv',index=False)
