import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
set_session(tf.Session(config=config))

from keras import backend as K
from keras.callbacks import Callback
import numpy as np
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
#categorical_crossentropy

K.set_image_dim_ordering('tf')
class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

def dump_history(logs):
    with open('train_loss','a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open('train_accuracy','a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open('valid_loss','a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open('valid_accuracy','a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

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
        return x_train[:22000],y_train[:22000],x_train[22000:],y_train[22000:],x_test

x_train,y_train,x_validate,y_validate,x_test=load_data()

model = Sequential()
model.add(Dense(units=256,input_dim=48*48,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=7,activation='softmax'))
model.summary()
#exit()
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
history = History()
model.fit(x_train,y_train,batch_size=100,epochs=100,validation_data=(x_validate,y_validate),callbacks=[history])
dump_history(history)
#model2.add(Dropout(0.25))
#model2.add(Conv2D(64,(3,3)))

#model2.add(Flatten())

#x_train = x_train.reshape(x_train.shape[0],48,48,1)
#x_validate = x_validate.reshape(x_validate.shape[0],48,48,1)
#x_test = x_test.reshape(x_test.shape[0],48,48,1)


score = model.evaluate(x_train,y_train)
print '\nTrain Acc:', score[1]
#score = model.evaluate(x_validate,y_validate)
#print '\nVal Acc:', score[1]


file_name = open('predict.csv','w')
res = model.predict_classes(x_test)

file_name.write("id,label\n")

for i in range(len(res)):
    file_name.write(str(i)+','+str(res[i])+'\n')
