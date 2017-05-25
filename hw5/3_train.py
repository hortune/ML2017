import re
from nltk.corpus import stopwords
import pickle
import sys,time
import numpy as np
# data 40220 unique after preprocess

def load_data(seperation=True):
    label_sets=[]
    text_sets=[]
    dic = {}
    dic1 = {}
    test_sets=[]
    for string in open(sys.argv[2],'r').readlines()[1:]:
        num,words=string.split(',',1)
        words = " ".join([word for word in words.split() if "http" not in word])
        # Remove all garbage punctuation and turn lower split
        words = re.sub("[^a-zA-Z]"," ",words).lower().split()
        # Remove stop words
        words = [w for w in words if w not in stopwords.words("english")]
        for i in words:
            dic[i] = True
        # Join back to string ?
        #test_sets.append( words if seperation else " ".join(words))            
        test_sets.append( words )            
    
    for string in open(sys.argv[1],'r').readlines()[1:]:
        num,label,words= string.split(',',2)
        words = " ".join([word for word in words.split() if "http" not in word])
        # Preprocess for label
        label_sets.append(label[1:-1].split())

        # Remove all garbage punctuation and turn lower split
        words = re.sub("[^a-zA-Z]"," ",words).lower().split()
        # Remove stop words
        for i in words:
            dic1[i] = True
        words = [w for w in words if w not in stopwords.words("english") and w in dic]
        # Join back to string ?
        text_sets.append( words if seperation else " ".join(words))            
    new_test_sets = []
    for string in test_sets:
        words = [word for word in string if word in dic1]
        new_test_sets.append( words if seperation else " ".join(words))            
    return np.array(text_sets),np.array(label_sets),new_test_sets
"""
# Save as pickle object
with open('data','wb') as f:
    pickle.dump(load_data(seperation=False),f)
"""
print("GG")
f = open('data','rb')
texts,labels,test_data = pickle.load(f)
"""
f2 = open('data_sep','rb')
t2, labels = pickle.load(f2)
"""
# =======================================

indices = np.arange(texts.shape[0])
np.random.shuffle(indices)
texts = texts[indices]
labels = labels[indices]

# =======================================

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC,SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

mul = MultiLabelBinarizer()
y_enc = mul.fit_transform(labels)

par =mul.get_params()
classifier = Pipeline([
    ('vectorizer',CountVectorizer(analyzer ="word", tokenizer = None, preprocessor = None, stop_words = None, max_features =30000)),
    ('tfidf',TfidfTransformer()),
    ('clf',OneVsRestClassifier(XGBClassifier(seed = 7122,scale_pos_weight=0.5)))])

classifier.fit(texts,y_enc)

with open('3_data','wb') as f:
    pickle.dump((classifier,mul),f)

predicted = classifier.predict(test_data)
with open('3','w') as fd:
    print("id,tags",file=fd)
    for index,text in enumerate(mul.inverse_transform(predicted)):
        print(index,",\""," ".join(text),"\"",sep='',file=fd)
# C= 1e-2 Eout = 49 random_state = 7122
