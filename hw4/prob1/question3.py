import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from math import sqrt
def load_data():
    res_img = []
    for i in range(ord('A'),ord('A')+10):
        for j in range(10):
            im = Image.open("dataset/"+chr(i)+"0"+str(j)+".bmp")
            p = np.array(im).flatten()
            res_img.append(p)
    return np.array(res_img,dtype="float64")
def rmse(res,origin):
    error = 0
    for i,j in zip(res,origin):
        error += (i-j).dot(i-j)
    return sqrt(error/(100*4096))/255

image_data = load_data()
mean = image_data.mean(axis=0)
image_data -= mean

covariance_matrix = np.cov(image_data.T)
eigval,eigvec = np.linalg.eigh(covariance_matrix) # eigh

idx = eigval.argsort()[::-1]

eigval = eigval[idx]
eigvec = eigvec.T
eigvec = eigvec[idx]
for i in range(1,101):
    res_image = []
    for row in eigvec[:i].dot(image_data.T).T:
        res_image.append( (row*eigvec[:i].T).sum(axis=1).reshape(4096))
    res_image= np.array(res_image)
    print ("error ",i,":",rmse(res_image,image_data))
