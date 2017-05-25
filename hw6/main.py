import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras


movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(
	keras.layers.Embedding(n_movies + 1, 32)(movie_input))
movie_vec = keras.layers.Dropout(0.5)(movie_vec)

user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(
	keras.layers.Embedding(n_users + 1, 32)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
nn = keras.layers.Dropout(0.5)(
	keras.layers.Dense(128, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(
	keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dense(128, activation='relu')(nn)

result = keras.layers.Dense(5, activation='softmax')(nn)

model = kmodels.Model([movie_input, user_input], result)
model.compile('adam', 'categorical_crossentropy'

