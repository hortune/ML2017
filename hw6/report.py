#coding = utf8
import pandas as pd
import numpy as np
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

import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from sklearn import cross_validation
def generate_model(n_movies, n_users):
        movie_input = keras.layers.Input(shape=[1])
        movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 32)(movie_input))
        movie_vec = keras.layers.Dropout(0.1)(movie_vec)

        user_input = keras.layers.Input(shape=[1])
        user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 32)(user_input))
        user_vec = keras.layers.Dropout(0.1)(user_vec)

        input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
        nn = keras.layers.Dropout(0.1)(keras.layers.Dense(128, activation='relu')(input_vecs))
        nn = keras.layers.normalization.BatchNormalization()(nn)
        nn = keras.layers.Dropout(0.1)(keras.layers.Dense(128, activation='relu')(nn))
        nn = keras.layers.normalization.BatchNormalization()(nn)
        nn = keras.layers.Dense(128, activation='relu')(nn)

        result = keras.layers.Dense(5, activation='softmax')(nn)

        model = kmodels.Model([movie_input, user_input], result)
        model.compile('adam', 'categorical_crossentropy')
        return model
if __name__ == '__main__':
    users, movies, ratings, test = load_data('users.csv','movies.csv','train.csv','test.csv')
    n_movies = movies.shape[0]
    n_users = users.shape[0]

    movieid = ratings.MovieID.cat.codes.values
    userid = ratings.UserID.cat.codes.values
    y = np.zeros((ratings.shape[0],5))
    y[np.arange(ratings.shape[0]), ratings.Rating -1] = 1
    
    m_test = test.MovieID.cat.codes.values
    u_test = test.UserID.cat.codes.values
    
    model =generate_model(n_movies, n_users)
    m_train,m_val,u_train,u_val,y_train,y_val = cross_validation.train_test_split(movieid,userid,y)
    model.fit([m_train,u_train],y_train,epochs=44,verbose=1,validation_data=([m_val,u_val],y_val),batch_size=40000)
    model.save('first.h5py')
    y_pred = np.argmax(model.predict([m_test,u_test]), 1)+1
    output = pd.DataFrame({'TestDataID':np.arange(len(y_pred)+1)[1:],'Rating':y_pred}, columns = ['TestDataID','Rating'])   
    output.to_csv('result.csv',index=False)

