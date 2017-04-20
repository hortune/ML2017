import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))
import numpy as np

from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
#categorical_crossentropy

def load_raw_data(name):
        file_name = open(name,'r')
        x = []
        y = []
        for line in file_name.readlines()[1:]:
            y.append(int(line.split(',')[0]))
            x.append(list(map(int,line.split(',')[1].split())))
        return np.array(x),np.array(y)

def load_data():
        x_train, y_train = load_raw_data('train.csv')
        x_test,y_test = load_raw_data('test.csv')
        #number = 10000   
        #x_train = x_train.reshape(x_train.shape[0], 48*48)
        #x_test = x_test.reshape(x_test.shape[0], 28*28)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, 7)
        #y_test = np_utils.to_categorical(y_test, 7)

        x_train = x_train/255
        x_test = x_test/255
	#x_test=np.random.normal(x_test)
        return x_train,y_train,x_train[20000:],y_train[20000:],x_test

x_train,y_train,x_validate,y_validate,x_test=load_data()

"""
model = Sequential()
model.add(Dense(input_dim=28*28,units=689,activation='relu'))
model.add(Dense(units=689,activation='relu'))
model.add(Dense(units=689,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)

score = model.evaluate(x_train,y_train)
print '\nTrain Acc:', score[1]
score = model.evaluate(x_test,y_test)
print '\nTest Acc:', score[1]
"""
model2 = Sequential()
model2.add(Conv2D(25,(3,3),input_shape=(48,48,1)))
model2.add(BatchNormalization())
#model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))

model2.add(Dropout(0.3))

model2.add(Conv2D(50,(3,3)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.3))

model2.add(Conv2D(100,(3,3)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(125,(3,3)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.5))

model2.add(Conv2D(250,(3,3)))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.5))
model2.add(Flatten())

model2.add(Dense(units=1000,activation='relu'))
model2.add(Dense(units=7,activation='softmax'))
model2.summary()

x_train = x_train.reshape(x_train.shape[0],48,48,1)
x_validate = x_validate.reshape(x_validate.shape[0],48,48,1)

x_test = x_test.reshape(x_test.shape[0],48,48,1)
model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

model2.fit(x_train,y_train,batch_size=100,epochs=100,validation_split=0.1)

score = model2.evaluate(x_train,y_train)
print '\nTrain Acc:', score[1]
score = model2.evaluate(x_validate,y_validate)
print '\nVal Acc:', score[1]


file_name = open('predict.csv','w')
res = model2.predict_classes(x_test)

file_name.write("id,label\n")

for i in range(len(res)):
    file_name.write(str(i)+','+str(res[i])+'\n')
