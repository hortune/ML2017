import re
from nltk.corpus import stopwords
import pickle
import sys,time
import numpy as np
# data 40220 unique after preprocess
from matplotlib import pyplot as plt
import operator
def load_data(seperation=True):
    dic = {}
    for string in open('train_data.csv','r').readlines()[1:]:
        num,label,words= string.split(',',2)
        for lab in label[1:-1].split():
            if lab in dic:
                dic[lab]+=1
            else:
                dic[lab]=1
    return dic

new_dic = sorted(load_data().items(), key=operator.itemgetter(1))
label,value = list(zip(*new_dic))


y_pos = np.arange(len(label))
locs, labels = plt.xticks(y_pos,label)
plt.setp(labels,rotation=90)
plt.bar(y_pos,value)
plt.show()
