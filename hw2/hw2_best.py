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
    
    """
    def fit(self,x,y):
        self.w_ = np.zeros(x.shape[1]) # make bias the zero one
        for i in range(0,self.n_iter):
                wt_x = np.dot(x,self.w_)*y
                ceta = 1./(np.ones(x.shape[0])+np.exp(wt_x))
                gradient = -np.dot(y.T*ceta.T,x)/(x.shape[0])
                self.w_-= self.eta*gradient/((gradient**2).sum())**0.5
                #E_in = np.log(1./ceta).sum()/x.shape[0]
                print("E_in",E_in)
        return self
    """
    def fit_hongyi(self,x,y):
        self.w_ = np.zeros(x.shape[1]) # make bias the zero one
        grad = np.ones(x.shape[1])
        for i in range(0,self.n_iter):
                wt_x = np.dot(x,self.w_)
                ceta = 1./(1+np.exp(-wt_x))
                
                
                gradient = np.dot((y-ceta).T,x)
                
                grad += gradient**2
                self.w_ += self.eta*(gradient/(grad**0.5))
                #self.eta*=0.95 
                #/((gradient**2).sum())**0.5
                #print(self.validate(x,y))
        return self
    
    def activation(self,x):
        return 1./(np.array([1])+np.exp(-np.dot(x,self.w_.T)))
    
    def validate(self,x,y):
        error = 0
        for x_,y_ in zip(x,y):
            if (abs(round(self.activation(x_)[0])-y_))>0.1:
                error+=1
        print("error : ",error)
        print (error/x.shape[0])
        return error/x.shape[0]

def shuffl(x,y,normal):
    index_shuf = [i for i in range(x.shape[0])]
    shuffle(index_shuf)
    list1=[]
    list2=[]
    for i in index_shuf:
        list1.append(x[i])
        list2.append(y[i])
    return np.array(list1),np.array(list2),normal

def load_training_data(filename,Y_filename,condition):
    x = genfromtxt(filename,delimiter=',')
    y = genfromtxt(Y_filename,delimiter=',')
    x = np.hstack((np.ones((x.shape[0],1),dtype=x.dtype),x))    
    x = np.delete(x,0,0)
    #x = np.delete(x,np.s_[20:],1)
    
    temp_x = []
    temp_y = []
    
    x = np.delete(x,[2],1)
    """
    for x_,y_  in zip(x,y):
        if y_ == 1:
            temp_x.append(x_)
            temp_x.append(x_)
            temp_x.append(x_)
            temp_y.append(y_)
            temp_y.append(y_)
            temp_y.append(y_)
   
    x = np.append(x,temp_x,axis=0)
    """
    normal = x.max(axis=0)
    x = x/normal
    #y = np.append(y,temp_y)
    
    return shuffl(x,y,normal)

def load_testing_data(filename,normal):
    x = genfromtxt(filename,delimiter=',')
    x = np.hstack((np.ones((x.shape[0],1),dtype=x.dtype),x)) 
    x = np.delete(x,0,0)
    x = np.delete(x,[2],1)
    x = x/normal
    return x
if __name__=='__main__':
    x,y,normal = load_training_data("X_train","Y_train",[])
    #exit()
    #print (y)
    #x = np.array([[1,0,0,0],[1,3,3,3],[1,4,4,4],[1,5,5,5]])
    #y = np.array([0,1,1,1])
    k = Regression(0.5,30000).fit_hongyi(x,y)
    
    test = load_testing_data("X_test",normal)
    with open("submission.csv","w+") as fd:
        print("id,label",file=fd) 
        for i,j in enumerate(test):
            print(str(i+1)+","+str(int(round(k.activation(j)[0],0))),file=fd)
