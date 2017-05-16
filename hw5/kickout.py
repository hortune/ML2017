import tensorflow as tf
import sys
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
import numpy as np

from keras.callbacks import Callback
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical
import keras.preprocessing.image as img
#categorical_crossentropy
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
import gensim
def f1score(y_true,y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1

def precision (y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true,y_pred):
    true_positives = K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
    recall = true_positives/(possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true,y_pred,beta =1):
    p = precision(y_true,y_pred)
    r = recall(y_true,y_pred)
    bb = beta**2
    fbeta_score = (1 + bb)*(p*r)/(bb*p+r+K.epsilon())
    return fbeta_score
def fmeasure(y_true,y_pred):
    return fbeta_score(y_true,y_pred,1)
def load_data():
    k = open('train_data.csv','r').readlines()
    x = []
    labels_index = {}
    labels = []
    for string in k[1:]:
        pre = string.find('"')
        last = string.find('"',pre+1)
        y_text  = string[pre+1:last].split()
        label = []
        for text in y_text:
            if text not in labels_index:
                lid = len(labels_index)
                labels_index[text] = lid
            label.append(labels_index[text])
        labels.append(label)
        x.append(string[last+2:-1])
    #print labels[:3]
    """
    y = open('test_data.csv','r').readlines()
    x_test = [] 
    for string in y[1:]:
        pre = string.find(',')
        x_test.append(string[pre+1:])
    """
    #x_test = tokenizer.texts_to_sequences(x_test)
    #x_test = pad_sequences(x_test,padding='post',maxlen=307)
    STOPWORDS = frozenset(['all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'fifty', 'four', 'not', 'own', 'through', 'yourselves', 'go', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'how', 'somewhere', 'with', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'under', 'ours', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very', 'de', 'none', 'cannot', 'every', 'whether', 'they', 'front', 'list','during', 'thus', 'now', 'him', 'nor', 'name', 'several', 'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone', 'either', 'each', 'become', 'thereupon', 'sometime', 'side', 'two', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up', 'namely', 'towards', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its', 'everything', 'behind', 'un', 'above','between', 'it', 'neither', 'seemed', 'ever', 'across', 'she', 'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere', 'although', 'found', 'alone', 're', 'along', 'fifteen', 'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others','line', 'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover', 'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due', 'been', 'next', 'anyone', 'eleven', 'much', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part','everywhere', 'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty', 'these', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'your', 'toward', 'my', 'something', 'and', 'whereafter', 'whenever', 'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps', 'latter','meanwhile', 'use', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon', 'eight', 'but', 'serious', 'nothing', 'such', 'why', 'a', 'off', 'whereby', 'third', 'i', 'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'whereas', 'once', 'using', 'does', 'did', 'didn', 'really', 'kg', 'regarding','unless', 'fify', 'say', 'km', 'used','various','just', 'quite', 'doing', 'don', 'doesn', 'make', 'file', 'creat', 'function', 'page', 'way', 'error', 'type', 'doe', 'custom', 'queri', 'data', 'work' , 'set', 'whi', 'applic', 'chang', 'add', 'multipl', 'best', 'code', 'server', 'user', 'view', 'differ', 'tabl', 'run', 'class', '2'])
    max_num = 0
    corpus = []
    for text in x:
        corpus.append([word for word in text_to_word_sequence(text,lower=True,split=" ") if word not in STOPWORDS])
        max_num = max(max_num,len(corpus[-1]))
    
    model = gensim.models.Word2Vec(corpus,size=50,window=8,min_count=0)
    x_res = []

    padding = np.zeros(50)
    for text in corpus:
        word_list= [] 
        for i in text:
            word_list.append(model[i])
        for k in range(max_num-len(text)):
            word_list.append(padding)
        x_res.append(word_list)
    data = np.array(x_res)
    # For shuffle
    #labels = to_categorical(np.array(labels))
    new_labels = []
    for i in labels:
        arr = np.zeros(len(labels_index))
        arr[i]=1
        new_labels.append(arr)
    labels = np.array(new_labels)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data,labels
#
x,labels = load_data()

m2 = Sequential()
m2.add(GRU(256,recurrent_dropout=0.2,inner_activation='hard_sigmoid',return_sequences=True,input_shape=(x.shape[1],50)))
m2.add(Dropout(0.3))
m2.add(GRU(256,recurrent_dropout=0.3,inner_activation='hard_sigmoid'))
m2.add(Dropout(0.4))
m2.add(Dense(689,activation='relu'))
m2.add(Dropout(0.5))
m2.add(Dense(38,activation='sigmoid'))
m2.summary()
m2.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=[fmeasure,precision,recall])
nb_validation_samples = int(0.2*x.shape[0])

x_train = x[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = x[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

m2.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_val,y_val))
res = m2.predict(x_val)
