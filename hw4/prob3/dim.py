from gen import gen_dim_with
import numpy as np
from math import log
test_data = np.load('/tmp2/data.npz')

def create_threstable ():
    dim_thres =[]
    dim_thres.append(0)
    for dim in range(1,61):
        thres = 0
        for i in range(10):
            data = gen_dim_with(dim)
            cov = np.cov(np.transpose(data))
            eig =  np.linalg.eig(cov)[0]
            thres += np.sum(eig[:dim])/np.sum(eig)
        dim_thres.append(thres/10)
    return dim_thres

threstable = create_threstable()
with open('ans.csv','w') as fd:
    print("SetId,LogDim",file=fd)
    for i in range(200):
        data = test_data[str(i)]
        cov = np.cov(np.transpose(data))
        eig =  np.linalg.eig(cov)[0]
        ans = 60
        for index in range(1,61):
            propo = np.sum(eig[:index])/np.sum(eig)
            if propo > threstable[index]:
                ans = index
                break
        print(i,",",log(ans),file=fd,sep='')
