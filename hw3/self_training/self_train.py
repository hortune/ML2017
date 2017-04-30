import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))
import numpy as np

from keras.callbacks import Callback
from numpy import genfromtxt
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import keras.preprocessing.image as img

#categorical_crossentropy

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
        x_train, y_train = load_raw_data('../train.csv')
        x_test,y_test = load_raw_data('../test.csv')
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = np_utils.to_categorical(y_train, 7)
        x_train = x_train/255
        x_test = x_test/255
        return x_train[:20000],y_train[:20000],x_train[20000:],y_train[20000:],x_test
def make_model():
        model2 = Sequential()
        model2.add(Conv2D(25,(3,3),input_shape=(48,48,1)))
        model2.add(BatchNormalization())
        model2.add(Dropout(0.3))
        model2.add(Conv2D(50,(3,3)))
        model2.add(Activation('relu'))
        model2.add(BatchNormalization())
        model2.add(MaxPooling2D((2,2)))
        model2.add(Dropout(0.3))
        model2.add(Conv2D(100,(3,3)))
        model2.add(Activation('relu'))
        model2.add(BatchNormalization())
        model2.add(MaxPooling2D((2,2)))
        model2.add(Dropout(0.3))
        model2.add(Conv2D(125,(3,3)))
        model2.add(Activation('relu'))
        model2.add(BatchNormalization())
        model2.add(MaxPooling2D((2,2)))
        model2.add(Dropout(0.3))
        model2.add(Conv2D(250,(3,3)))
        model2.add(Activation('relu'))
        model2.add(BatchNormalization())
        model2.add(MaxPooling2D((2,2)))
        model2.add(Dropout(0.4))
        model2.add(Flatten())
        model2.add(Dense(units=1000,activation='relu'))
        model2.add(Dense(units=7,activation='softmax'))
        model2.summary()
        model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        return model2



x_train,y_train,x_validate,y_validate,x_test=load_data()
x_train = x_train.reshape(x_train.shape[0],48,48,1)
x_validate = x_validate.reshape(x_validate.shape[0],48,48,1)
x_test = x_test.reshape(x_test.shape[0],48,48,1)


model=load_model('../last_model') # load model
res = model.predict_classes(x_test)
x_train= np.vstack((x_train, x_test))
y_test = np_utils.to_categorical(res, 7)
y_train = np.vstack((y_train,y_test))
del model

model2 = make_model()

datagen = img.ImageDataGenerator(
	rotation_range = 3,
	horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)
datagen.fit(x_train)

history = History()
model2.fit_generator(datagen.flow(x_train,y_train,batch_size=128),steps_per_epoch=len(x_train)/16,epochs=300,validation_data=(x_validate,y_validate),callbacks=[history])

dump_history(history)


score = model2.evaluate(x_train,y_train)
print '\nTrain Acc:', score[1]
score = model2.evaluate(x_validate,y_validate)
print '\nVal Acc:', score[1]
model2.save('self_training_model')

file_name = open('last.csv','w')
res = model2.predict_classes(x_test)
file_name.write("id,label\n")

for i in range(len(res)):
    file_name.write(str(i)+','+str(res[i])+'\n')
