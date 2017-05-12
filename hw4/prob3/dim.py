from gen import gen_dim_with
import numpy as np
from math import log
test_data = np.load('/tmp2/data.npz')
"""
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
"""
#threstable = create_threstable()
#print(threstable)
threstable= [0, 0.87543429634510428, 0.85254222399050461, 0.85435788931559864, 0.83149856199215899, 0.83027505350659447, 0.82536861535087525, 0.82268718359107229, 0.81289014174251828, 0.82146743034213388, 0.82028850196821568, 0.81884721595875065, 0.82536494286032358, 0.82814989875908063, 0.83301111261472938, 0.83011985242306463, 0.84330958143446444, 0.84255020969852323, 0.84423911659838891, 0.85395144947085611, 0.85700914551121099, 0.86086364433432883, 0.8668451675194252, 0.86844412098854118, 0.87424369021729975, 0.87564331937262863, 0.88468287141113355, 0.88885971599331715, 0.89618986408277446, 0.89762210556714928, 0.90184948983329227, 0.90477670707629976, 0.91313677718438657, 0.91463118808208799, 0.91824132494404576, 0.92432580046705171, 0.92464663273172731, 0.92769128651040322, 0.9304295605055326, 0.93793554572872451, 0.93811466165496216, 0.94244798782506756, 0.9456673532175911, 0.94766145933308699, 0.95116277459908649, 0.95377687067418893, 0.95698741127552811, 0.95906295407222208, 0.96206426669716372, 0.96360841812038056, 0.9658219953563032, 0.96761375219092582, 0.97090042867915827, 0.97071546073215775, 0.97327082816788091, 0.9762007678328104, 0.9767608921507881, 0.97884033641229196, 0.97989273417249656, 0.98144866005081821, 0.98288768237429702]

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
