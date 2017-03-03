# coding: utf-8
import numpy as np
from numpy import *
import math

class Regression(object):
    def __init__ (self, eta=1e-20, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self,x,y):
        self.w_ = np.zeros(1 + 18*9) # make bias the zero one
        for i in range(0,self.n_iter):
            output = np.dot(x,self.w_[1:])+self.w_[0]
            errors = y- output 
            self.w_[1:]+=self.eta*x.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
        return self
    def activation(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]

data=genfromtxt('train.csv',delimiter=',')
for i in range(0,data.shape[0]):
    for j in range(0,data.shape[1]):
        if math.isnan(data[i][j]):
            data[i][j]=0
data=np.delete(np.delete(data,0,0),np.s_[0:3],1)
data=np.delete(data,np.s_[10:],1)
data=np.split(data,240)


y=np.array([ i[9][9] for i in data])
#x=np.array([ i[9][0:9] for i in data])
data=np.delete(data,np.s_[9:],2)
x=np.array([ np.ravel(i) for i in data])
k=Regression(1e-9,200000).fit(x,y)
total_data=240
total_true=0


data=genfromtxt('test_X.csv',delimiter=',')
for i in range(0,data.shape[0]):
    for j in range(0,data.shape[1]):
        if math.isnan(data[i][j]):
            data[i][j]=0
data=np.delete(data,np.s_[0,2],1)
data=np.split(data,240)
qq=np.array([ np.ravel(i) for i in data])
total_true=0
for (a,b) in zip (x,y):
    if int(k.activation(a)) == b:
        total_true = total_true+1
print (total_true/240.)
with open('submission.csv',"w+") as fd:
    print("id,value",file=fd) 
    for i in range(0,240):
        print("id_"+str(i)+","+str(int(k.activation(qq[i]))),file=fd)
#try int and round
