import sys, csv, string
#  from IPython import embed
import nltk
import random
import numpy as np
import scipy.stats
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from pandas import *
import re
import pickle
import math
import copy
seed = 7122#int(sys.argv[4])
random.seed(seed)
np.random.seed(seed)

filtered = (
        ['\'s'] +
        list(string.punctuation) +
        list(map(lambda x: x*2, string.punctuation))
    )
def loadData(path, test=False):
    if True:
        data = []
        yearRegex = re.compile(r'^([0-9]{4})s?(-([0-9]{4})s?)?$')
        urlRegex = re.compile(r'(https?:\/\/)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
        dotRegex = re.compile(r'\.+')
        yearRegex2 = re.compile(r'(century|year)')
        htmlRegex = re.compile(r'&[a-zA-Z]{,6};')
        ageRegex = re.compile(r'(year-old|[0-9]+-year)')
        numRegex = re.compile(r'^[0-9]+(\,[0-9]*)*(\.[0-9]*)?-?(st|nd|rd|th)?(-?[0-9]+(\,[0-9]*)*(\.[0-9]*)?-?(st|nd|rd|th)?)?$')
        numRegex2 = re.compile(r'^((thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)-?)?((eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)-?)?((one|two|three|four|five|six|seven|eight|nine|ten)-?)?((first|second|third|forth|fifth|sixth|seventh|eighth|nineth|tenth)-?)*$')
        numRegex3 = re.compile(r'([0-9]+-?)')
        numRegex4 = re.compile(r'((thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third|forth|fifth|sixth|seventh|eighth|nineth|tenth)-?)+')
        langRegex = re.compile(r'\s[a-z]{2}:')
        with open(path, 'r') as f:
            f.readline()
            for line in f:
                if test:
                    ig, corpus = line.split(',', 1)
                    tags = []
                else:
                    ig, tags, corpus = line.split(',', 2)
                    tags = tags.replace('"', '').split(' ')
                corpus = urlRegex.sub('', corpus)
                corpus = corpus.lower()
                corpus = corpus.replace('‘', '\'')
                corpus = corpus.replace('‚', ',')
                corpus = corpus.replace('‛', '\'')
                corpus = corpus.replace('“', '\'')
                corpus = corpus.replace('”', '\'')
                corpus = corpus.replace('„', ' ')
                corpus = corpus.replace('‟', '\'')
                corpus = corpus.replace('•', ' ')
                corpus = corpus.replace('’', '\'')
                corpus = corpus.replace('—', '-')
                corpus = corpus.replace('…', '.')
                corpus = corpus.replace('=', '')
                corpus = dotRegex.sub('.', corpus)
                corpus = htmlRegex.sub(' ', corpus)
                corpus = langRegex.sub(' ', corpus)
                corpus = corpus.replace('/', ' ')
                corpus = corpus.replace('-', ' ')
                corpus = corpus.replace(':', ' ')
                corpus = corpus.replace('\'', ' ')
                corpus = nltk.word_tokenize(corpus)
                corpus = list(filter(lambda x: x not in filtered, corpus))
                def preprocess(word):
                    if word.startswith('\''):
                        word = word[1:]
                    if word.startswith('*'):
                        word = word[1:]
                    if word.startswith('-'):
                        word = word[1:]
                    if word.endswith('-'):
                        word = word[:-1]
                    m = yearRegex.match(word)
                    if m:
                        return "YEAR"
                    m = yearRegex2.search(word)
                    if m:
                        return "YEAR"
                    m = ageRegex.search(word)
                    if m:
                        return "AGE"
                    m = numRegex.match(word)
                    if m:
                        return "NUMBER"
                    m = numRegex2.match(word)
                    if m:
                        return "NUMBER"
                    suffix = ['es', 's', 'nes', 'hood', 'ship', 'ment', 'ed', 'ically', 'ical', 'ion', 'ing', 'ability', 'ible', 'ish', 'ian', 'ier', 'er', 'ee', 'ity', 'ty', 'ily', 'ly', 'y', 'e', 'ing']
                    for _ in range(2):
                        for s in suffix:
                            if word.endswith(s):
                                word = word[:-len(s)]
                    word = numRegex3.sub('', word)
                    word = numRegex4.sub('', word)
                    return word
                corpus = list(map(preprocess, corpus))
                #  corpus = list(set(corpus))
                data.append([ig, tags, corpus])
        data = DataFrame(data, columns=['id', 'tags', 'corpus']).set_index('id')
        with open(path+'.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(path+'.pkl', 'rb') as f:
            data = pickle.load(f)
    return data

def vectorize(train, test, mask):
    voclist = np.unique(np.concatenate(mask['corpus'].values))
    voc = {}
    for i, w in enumerate(voclist):
        voc[w] = i
    train['corpus'] = train['corpus'].apply(lambda x: [voc[e] for e in x if e in voc])
    test['corpus'] = test['corpus'].apply(lambda x: [voc[e] for e in x if e in voc])
    return train, test, voclist

train = loadData(sys.argv[1])
test = loadData(sys.argv[2], test=True)
data = train
data, test, voclist = vectorize(data, test, test) #valid, test)
veclen = len(voclist)
testid = list(test.index)

voc = np.zeros(veclen)
taglist = np.unique(np.concatenate(data['tags'].values))
freq = dict()
for tag in taglist:
    freq[tag] = {
        'tp': np.zeros(veclen),
        'fp': np.zeros(veclen),
        'tn': np.zeros(veclen),
        'fn': np.zeros(veclen),
    }

for tags, corpus in data.values:
    c = np.zeros(veclen)
    for w in corpus:
        c[w] = 1
    voc += c
    for tag in taglist:
        if tag in tags:
            freq[tag]['tp'] += c
            freq[tag]['fn'] += 1-c
        else:
            freq[tag]['fp'] += c
            freq[tag]['tn'] += 1-c

for tag in freq:
    v = freq[tag]
    tp = v['tp'] / (v['tp'] + v['fp'])
    tr = v['tp'] / (v['tp'] + v['fn'])
    tf = 2 * tp * tr / (tp + tr)
    tf = np.nan_to_num(tf)
    fp = v['fp'] / (v['fp'] + v['tp'])
    fr = v['fp'] / (v['fp'] + v['tn'])
    ff = 2 * fp * fr / (fp + fr)
    ff = np.nan_to_num(ff)

    dif = (voc - v['tp']) / len(taglist)
    dif += (voc == v['tp']) * (1/100)
    dif += (dif == 0) * 1
    freq[tag] = v['tp'] / dif
    v = freq[tag]
    vf = sorted(enumerate(v), key=lambda x: x[1], reverse=True)

def normalize(freq, train, test):
    ctr = np.zeros(veclen)
    vtr = np.concatenate(train['corpus'].values)
    for w in vtr:
        ctr[w] += 1
    cte = np.zeros(veclen)
    vte = np.concatenate(test['corpus'].values)
    for w in vte:
        cte[w] += 1
    ctr += (ctr == 0)*1
    weight = (cte / ctr)**0.025
    for tag in freq:
        freq[tag] *= weight

def score(data):
    ret = {}
    hit = 0
    nothit = 0
    for tag in freq:
        b = []
        v = freq[tag]
        for tags, corpus in data.values:
            ss = np.take(v, corpus)
            ss = ss[ss!=0]
            hit += len(ss)
            nothit += len(corpus) - len(ss)
            #  s = np.mean(ss)
            s = scipy.stats.gmean(ss)
            if tag not in tags:
                b.append([0,s])
            else:
                b.append([1,s])
        ret[tag] = np.array(b)
    return ret, hit / (hit+nothit)
baseline, _ = score(data)
testScore, vHitRate = score(test)

def f1(ans, pred):
    eq = (ans == pred)
    ne = (ans != pred)
    po = (pred == 1)
    ng = (pred == 0)
    tp = np.sum(np.logical_and(eq, po))
    fp = np.sum(np.logical_and(ne, po))
    fn = np.sum(np.logical_and(ne, ng))
    print("tp: %d, fp: %d, fn: %d" %(tp,fp,fn))
    if tp == 0:
        return 0
    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    return 2 * pr * re / (pr + re)

class NaiveCut():
    def fit(self, x, y):
        x.reshape((-2,))
        x0 = x[y==0]
        x1 = x[y==1]
        x1 = scipy.stats.gmean(x1)
        x0 = scipy.stats.gmean(x0)
        self.baseline = scipy.stats.gmean([x0, x1])
    def predict(self, x):
        return (x.reshape((-2,)) > self.baseline) * 1
        
res = {}
for tid in testid:
    res[tid] = []

apy = []
avy = []
for tag in freq:
    m = NaiveCut()
    x = baseline[tag][:,1:]
    vx = testScore[tag][:, 1:]
    y = baseline[tag][:,0]
    vy = testScore[tag][:, 0]
    #  m = GradientBoostingClassifier(max_depth=10)
    m.fit(x, y)
    pv = m.predict(vx)
    for i, e in zip(testid, pv):
        if e:
            res[i].append(tag)
    avy = avy + vy.tolist()
    apy = apy + pv.tolist()
    acc = f1(y, m.predict(x))
    vacc = f1(vy, pv)
    baseline[tag] = m

apy = np.array(apy)
avy = np.array(avy)
af1 = f1(avy, apy)

f = open("0", 'w')
f.write('"id","tags"\n')
for tid in testid:
    f.write('"%s","%s"\n' % (tid, ' '.join(res[tid])))
f.close()

