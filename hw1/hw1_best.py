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
        print (cost/len(x))
        return self
    def validate(self,x,y):
        errors = 0
        for x_,y_ in zip(x,y):
            errors+=(self.activation(x_)-y_)**2
        errors/=len(y)
        return errors**0.5

def sample(data):
    res = []
    for k in data:
        d = []
        for i in k:
            for j in range(0,9):
                d.append(i[j])
        res.append(d)
    return np.array(res)
def increase_data(data):
    data = np.split(data,12)
    ret = []
    ans = [] 
    for mon in data:
        date_data = np.split(mon,20)
        mo = []
        for i in range(0,18):
            mo.append([])
        for i in date_data:
            for j in range(0,18):
                mo[j]+=list(i[j])
        i=0
        while (i+10)<480:
            simple=[]
            for q in range(0,18):
                simple.append(mo[q][i:i+9])
            ret.append(simple)
            ans.append(mo[9][i+9])
            i+=1
    print ("first",len(ret),"second",len(ret[0]),"third",len(ret[0][0]))
    return np.array(ret),np.array(ans)
def load_training_data(filename):
    data=genfromtxt(filename,delimiter=',')
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if math.isnan(data[i][j]):
                data[i][j]=0
    data=np.delete(np.delete(data,0,0),np.s_[0,3],1)
    data,y = increase_data(data)
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
    return sample(data)

x,y= load_training_data('train.csv')
test_data=load_testing_data('test_X.csv')
delta = 2,500
for i in range(0,50):
    print ("learning rate",init)
    k=Regression(init,10000).fit_adagrad(x[0:4512],y[0:4512])
    init/=delta
    print("rmse",k.validate(x[4512:],y[4512:]))
with open('submission.csv',"w+") as fd:
    print("id,value",file=fd) 
    for i in range(0,240):
        print("id_"+str(i)+","+str(k.activation(test_data[i])),file=fd)
