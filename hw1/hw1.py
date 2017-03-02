# coding: utf-8
import numpy as np
from numpy import *

"""
manipulate training data x,y
"""
data=genfromtxt('train.csv',delimiter=',')
data=np.delete(np.delete(data,0,0),np.s_[0:3],1)
data=np.delete(data,np.s_[10:],1)
data=np.split(data,240)
x=[ i[9][0:9] for i in data]
y=[ i[9][9] for i in data]


