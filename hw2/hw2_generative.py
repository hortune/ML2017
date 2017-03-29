# coding: utf-8
import numpy as np
from numpy import *
import math
import sys
from random import shuffle


class Generative(object):
    def guassian_mean_sigma(self,c1):
        mean = c1.sum(axis=0,dtype=np.float64)/c1.shape[0]
        x_minus_mean = c1-mean
        x_minus_mean = x_minus_mean.sum(axis=0,dtype=np.float64)/c1.shape[0]
        sigma = np.outer(x_minus_mean,x_minus_mean)
        return mean,sigma
    
    def guassian_model(self,c1,c2):
        self.true_mean,true_sigma = self.guassian_mean_sigma(c1)
        self.false_mean, false_sigma = self.guassian_mean_sigma(c2)
        
        self.rate = c1.shape[0]/(c2.shape[0]+c1.shape[0])
        self.sigma = true_sigma*self.rate + (1-self.rate)*false_sigma
        self.inverse = np.linalg.inv(self.sigma)
        return self       

    def activate(self,x):
        true_x = x-self.true_mean
        true_numerator = -(true_x.dot(self.inverse)*true_x).sum(axis=1)/2

        false_x = x-self.false_mean
        false_numerator = -(false_x.dot(self.inverse)*true_x).sum(axis=1)/2
        
        numerator = false_numerator - true_numerator
        print(numerator)
        #ans = 1/(1+np.exp(numerator)*self.rate/(1-self.rate))
        #print(ans[0])
def load_training_data(filename,Y_filename):
    x = genfromtxt(filename,delimiter=',',dtype=np.float64)
    y = genfromtxt(Y_filename,delimiter=',')
    x = np.delete(x,0,0)
    
    """
    normalize prevent underflow
    """
    mean = x.mean(axis=0,dtype=np.float64)
    x -= mean
    sigma = (x**2).mean(axis=0,dtype=np.float64)/x.shape[0]
    x /= sigma
    
    
    true_data=[]
    false_data=[]

    for q,p in zip(x,y):
        if p == 0:
            false_data.append(q)
        else:
            true_data.append(q)
    return np.array(true_data),np.array(false_data),mean,sigma

def load_testing_data(filename,normal,s1,s2):
    x = genfromtxt(filename,delimiter=',')
    x = np.delete(x,0,0)
    x = x/normal
    return x
    
if __name__=='__main__':
    true_data,false_data,mean,sigma = load_training_data("X_train","Y_train")
    k = Generative().guassian_model(true_data,false_data)
    k.activate(true_data)
    #test = load_testing_data("X_test",normal,0,6)
    """
    with open("submission.csv","w+") as fd:
        print("id,label",file=fd) 
        for i,j in enumerate(test):
            print(str(i+1)+","+str(int(round(k.activation(j)[0],0))),file=fd)
    """
