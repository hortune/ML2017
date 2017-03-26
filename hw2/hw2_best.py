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
    
    def fit(self,x,y):
        self.w_ = np.zeros(x.shape[1]) # make bias the zero one
        for i in range(0,self.n_iter):
                wt_x = np.dot(x,self.w_)*y
                ceta = 1./(np.ones(x.shape[0])+np.exp(wt_x))
                gradient = np.dot(y.T*ceta.T,x)/(x.shape[0])
                self.w_-= self.eta*gradient/((gradient**2).sum())**0.5
                E_in = np.log(1./ceta).sum()/x.shape[0]
                print("E_in : ",E_in)
        return self
    def activation(self,x):
        return 1./(np.ones(x.shape[0])+np.exp(-np.dot(x,self.w_)))
    
    def validate(self,x,y):
        error = 0
        for x_,y_ in zip(x,y):
            print (self.activation(x_)[0])
        return error

def shuffl(x,y):
    index_shuf = [i for i in range(x.shape[0])]
    shuffle(index_shuf)
    list1=[]
    list2=[]
    for i in index_shuf:
        list1.append(x[i])
        list2.append(y[i])
    return np.array(list1),np.array(list2)

def load_training_data(filename,Y_filename,condition):
    x = genfromtxt(filename,delimiter=',')
    y = genfromtxt(Y_filename,delimiter=',')
    x = np.hstack((np.ones((x.shape[0],1),dtype=x.dtype),x))    
    x = np.delete(x,0,0)
    x = np.delete(x,np.s_[20:],1)
    
    temp_x = []
    temp_y = []
    for x_,y_  in zip(x,y):
        if y_ == 1:
            temp_x.append(x_)
            temp_x.append(x_)
            temp_x.append(x_)
            temp_y.append(y_)
            temp_y.append(y_)
            temp_y.append(y_)
    x = np.append(x,temp_x,axis=0)
    y = np.append(y,temp_y)
    return shuffl(x,y)
if __name__=='__main__':
    #x,y = load_training_data("X_train","Y_train",[])
    #print (y)
    x = np.array([[1,2,3],
        [1,2,3],
        [2,3,4],
        [5,6,8]])
    y = np.array([1,1,1,0])
    k = Regression(1e-10,100).fit(x,y)
    print( k.validate(x,y))    
