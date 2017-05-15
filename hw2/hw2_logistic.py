# coding: utf-8
import numpy as np
from numpy import *
import math
import sys
from random import shuffle

class Regression(object):
    def __init__ (self, eta=1e-20, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
    
    def clean_weight(self):
        for i in range(self.w_.shape[0]):
            if abs(self.w_[i])<1e-4:
                self.w_[i]=0
    
    def fit_hongyi(self,x,y,lam=0):
        self.w_ = np.zeros(x.shape[1]) # make bias the zero one
        grad = np.ones(x.shape[1])
        for i in range(0,self.n_iter):
                wt_x = np.dot(x,self.w_)
                ceta = 1./(1+np.exp(-wt_x))
                gradient = np.dot((y-ceta).T,x)
                gradient += lam*(self.w_/x.shape[0])
                grad += gradient**2
                self.w_ += self.eta*(gradient/(grad**0.5))
        return self
    
    def activation(self,x):
        return 1./(np.array([1])+np.exp(-np.dot(x,self.w_.T)))
    
    def validate(self,x,y):
        error = 0
        for x_,y_ in zip(x,y):
            if (abs(round(self.activation(x_)[0])-y_))>0.1:
                error+=1
        return error/x.shape[0]

def shuffl(x,y,mean,std):
    index_shuf = [i for i in range(x.shape[0])]
    shuffle(index_shuf)
    list1=[]
    list2=[]
    for i in index_shuf:
        list1.append(x[i])
        list2.append(y[i])
    return np.array(list1),np.array(list2),mean,std

def sample(x,i,j):
    #x = np.hstack((x,x[:,i:j]**0.5))
    x = np.hstack((x,x[:,i:j]**2))
    x = np.hstack((x,x[:,i:j]**3))
    x = np.hstack((x,x[:,i:j]**4))
    return x

def load_training_data(filename,Y_filename,s1,s2):
    x = genfromtxt(filename,delimiter=',')
    y = genfromtxt(Y_filename,delimiter=',')
    x = np.delete(x,0,0)
    
    temp_x = []
    temp_y = []
    
    x = np.delete(x,[1],1)
    
    std = x.max(axis=0)
    x = x/std
    
    mean = x.mean(axis =0)
    #std = x.std(axis = 0)
    #x = (x - mean)/std
    x = sample(x,s1,s2)
    x = np.hstack((np.ones((x.shape[0],1),dtype=x.dtype),x))    
    return shuffl(x,y,mean,std)

def load_testing_data(filename,mean,std,s1,s2):
    x = genfromtxt(filename,delimiter=',')
    #x = np.hstack((np.ones((x.shape[0],1),dtype=x.dtype),x)) 
    x = np.delete(x,0,0)
    x = np.delete(x,[1],1)
    x /= std
    x = sample(x,s1,s2)
    x = np.hstack((np.ones((x.shape[0],1),dtype=x.dtype),x)) 
    return x
    
if __name__=='__main__':
    x,y,mean,std = load_training_data(sys.argv[1],sys.argv[2],0,6)
    k = Regression(0.5,30000).fit_hongyi(x[:],y[:],lam=0)
    test = load_testing_data(sys.argv[3],mean,std,0,6)
    with open(sys.argv[4],"w+") as fd:
        print("id,label",file=fd) 
        for i,j in enumerate(test):
            print(str(i+1)+","+str(int(round(k.activation(j)[0],0))),file=fd)
