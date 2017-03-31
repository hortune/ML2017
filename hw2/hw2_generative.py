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
        self.rate = c1.shape[0]/(c2.shape[0]+c1.shape[0])
        self.sigma = true_sigma*self.rate + (1-self.rate)*false_sigma
        self.inverse = np.linalg.inv(self.sigma)
        return self       

    def activate(self,x):
        w = (self.true_mean-self.false_mean).dot(self.inverse).dot(x.T)
        b1 = -(self.true_mean).dot(self.inverse).dot(self.true_mean.T)/2
        b2 = (self.false_mean).dot(self.inverse).dot(self.false_mean.T)/2
        b3 = math.log((self.rate)/(1-self.rate))
        ans = 1/(1+np.exp(-w-b1-b2-b3))
        return ans
    
    def validate(self,x,y):
        ans = self.activate(x)
        error = 0
        for i,j in zip(ans,y):
            if round(i,0) != j:
                error += 1
        return error/x.shape[0]
                

def load_training_data(filename,Y_filename):
    x = genfromtxt(filename,delimiter=',',dtype=np.float64)
    y = genfromtxt(Y_filename,delimiter=',')
    x = np.delete(x,0,0)
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
    return np.array(true_data),np.array(false_data),mean,std

def load_testing_data(filename,mean,std):
    x = genfromtxt(filename,delimiter=',')
    x = np.delete(x,0,0)
    x = (x-mean)/std
    return x
    
if __name__=='__main__':
    true_data,false_data,mean,std = load_training_data(sys.argv[1],sys.argv[2])
    k = Generative().guassian_model(true_data,false_data)
    test_data = load_testing_data(sys.argv[3],mean,std)
    test_result = k.activate(test_data)
    
    with open(sys.argv[4],"w+") as fd:
        print("id,label",file=fd) 
        for i,j in enumerate(test_result):
            print(str(i+1)+","+str(int(round(j,0))),file=fd)
