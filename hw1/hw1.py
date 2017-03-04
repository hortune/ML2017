# coding: utf-8
# TODO SGD or Momentum
# 6 8 9 10 11 
import numpy as np
from numpy import *
import math

class Regression(object):
    def __init__ (self, eta=1e-20, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self,x,y):
        self.w_ = np.zeros(1 + x.shape[1]) # make bias the zero one
        cost=1
        x_mon = np.split(x,12)
        y_mon = np.split(y,12)
        for i in range(0,self.n_iter):
                output = np.dot(x,self.w_[1:])+self.w_[0]
                errors = y- output
                self.w_[1:]+=self.eta*x.T.dot(errors)
                self.w_[0]+=self.eta*errors.sum()
                loss = errors
        print ((errors**2).sum())
        return self
    def activation(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]
    def fit_adagrad(self,x,y):
        self.w_ = np.zeros(1 + x.shape[1]) # make bias the zero one
        cost,k,g=1,0,0
        for i in range(0,self.n_iter):
                output = np.dot(x,self.w_[1:])+self.w_[0]
                errors = y- output
                grad = x.T.dot(errors)
                k+= grad**2
                g+= (errors.sum())**2
                self.w_[1:]+=self.eta*grad/(k**0.5)
                self.w_[0]+=self.eta*errors.sum()/(g**0.5)
                cost=np.sum(errors**2)
        print (cost)
        return self

def sample(data):
    res = []
    for k in data:
        d = []
        for i in k:
            for j in range(0,9):
                d.append(i[j])
        res.append(d)
    return np.array(res)
def load_training_data(filename):
    data=genfromtxt(filename,delimiter=',')
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if math.isnan(data[i][j]):
                data[i][j]=0
    data=np.delete(np.delete(data,0,0),np.s_[0:3],1)
    data=np.delete(data,np.s_[10:],1)
    data=np.split(data,240)
    y=np.array([ i[9][9] for i in data])
    #data=np.delete(data,np.s_[9:],2)
    #x=np.array([ np.ravel(i) for i in data])
    x= sample(data)
    return (x,y)
def load_testing_data(filename):
    data=genfromtxt(filename,delimiter=',')
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if math.isnan(data[i][j]):
                data[i][j]=0
    data=np.delete(data,np.s_[0,2],1)
    data=np.split(data,240)
    #return np.array([ np.ravel(i) for i in data])
    return sample(data)
x,y= load_training_data('train.csv')
test_data=load_testing_data('test_X.csv')
delta = 2
init = 500
for i in range(0,50):
    print ("learning rate",init)
    k=Regression(init,1000000).fit_adagrad(x,y)
    break
    init/=delta
total_data=240
total_true=0
for (a,b) in zip (x,y):
    if int(k.activation(a)) == b:
        total_true = total_true+1
with open('submission.csv',"w+") as fd:
    print("id,value",file=fd) 
    for i in range(0,240):
        print("id_"+str(i)+","+str(k.activation(test_data[i])),file=fd)
