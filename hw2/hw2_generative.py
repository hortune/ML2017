# coding: utf-8
import numpy as np
from numpy import *
import math
import sys
from random import shuffle


class Generative(object):
    def guassian_mean_sigma(self,c1):
        mean = c1.mean(axis=0,dtype=np.float64)
        x_minus_mean = c1-mean
        #x_minus_mean = x_minus_mean.sum(axis=0,dtype=np.float64)/c1.shape[0]
        
        sigma = 0
        for row in x_minus_mean:
            sigma += np.outer(row,row)
        sigma /= c1.shape[0]
        return mean,sigma
    
    def guassian_model(self,c1,c2):
        self.true_mean,true_sigma = self.guassian_mean_sigma(c1)
        self.false_mean, false_sigma = self.guassian_mean_sigma(c2)
        #print(true_sigma[0])
        self.rate = c1.shape[0]/(c2.shape[0]+c1.shape[0])
        self.sigma = true_sigma*self.rate + (1-self.rate)*false_sigma
        self.inverse = np.linalg.inv(self.sigma)
        return self       
    """
    def activate(self,x):
        true_x = x-self.true_mean
        true_numerator = -(true_x.dot(self.inverse)*true_x).sum(axis=1)/2

        false_x = x-self.false_mean
        false_numerator = -(false_x.dot(self.inverse)*true_x).sum(axis=1)/2
        
        numerator = false_numerator - true_numerator
        ans = 1/(1+np.exp(numerator)*self.rate/(1-self.rate))
        return ans
    """

    def new_activate(self,x):
        w = (self.true_mean-self.false_mean).dot(self.inverse).dot(x.T)
        #print(w)
        b1 = -(self.true_mean).dot(self.inverse).dot(self.true_mean.T)/2
        b2 = (self.false_mean).dot(self.inverse).dot(self.false_mean.T)/2
        b3 = math.log((self.rate)/(1-self.rate))
        ans = 1/(1+np.exp(-w-b1-b2-b3))
        return ans
    
    def validate(self,x,y):
        ans = self.new_activate(x)
        error = 0
        for i,j in zip(ans,y):
            if round(i,0) != j:
                error += 1
        return error/x.shape[0]
                

def load_training_data(filename,Y_filename):
    x = genfromtxt(filename,delimiter=',',dtype=np.float64)
    y = genfromtxt(Y_filename,delimiter=',')
    x = np.delete(x,0,0)
    """
    x = np.hstack((x,x[:,0:6]**3/1e-11))
    x = np.hstack((x,x[:,0:6]**0.5/1e-3))
    x = np.hstack((x,x[:,0:6]**1.5/1e-18))
    x = np.hstack((x,x[:,0:6]**2/1e-15))
    x = np.hstack((x,x[:,0:6]**2.5/1e-20))
    """
    mean = x.mean(axis=0,dtype=np.float64)
    x -= mean
    #sigma = np.sqrt((x**2).mean(axis=0,dtype=np.float64)/x.shape[0])
    std = x.std(axis=0)
    x /= std
    true_data=[]
    false_data=[]
    for q,p in zip(x,y):
        if p == 0:
            false_data.append(q)
        else:
            true_data.append(q)
    return x,y,np.array(true_data),np.array(false_data),mean,std

def load_testing_data(filename,normal,s1,s2):
    x = genfromtxt(filename,delimiter=',')
    x = np.delete(x,0,0)
    x = x/normal
    return x
    
if __name__=='__main__':
    x,y,true_data,false_data,mean,sigma = load_training_data("X_train","Y_train")
    #true_data = np.array([[1,0],[0,1]])
    #false_data = np.array([[1,2],[3,4]])
    k = Generative().guassian_model(true_data,false_data)
    print(k.validate(x,y))
    #test = load_testing_data("X_test",normal,0,6)
    """
    with open("submission.csv","w+") as fd:
        print("id,label",file=fd) 
        for i,j in enumerate(test):
            print(str(i+1)+","+str(int(round(k.activation(j)[0],0))),file=fd)
    """
