import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_data():
    res_img = []
    for i in range(ord('A'),ord('A')+10):
        for j in range(10):
            im = Image.open("dataset/"+chr(i)+"0"+str(j)+".bmp")
            p = np.array(im).flatten()
            res_img.append(p)
    return np.array(res_img,dtype="float64")

image_data = load_data()
mean = image_data.mean(axis=0)
image_data -= mean

covariance_matrix = np.cov(image_data.T)
eigval,eigvec = np.linalg.eigh(covariance_matrix) # eigh

idx = eigval.argsort()[::-1]

eigval = eigval[idx]
eigvec = eigvec.T
eigvec = eigvec[idx]



fig = plt.figure(figsize=(12,12))
for i in range(100):
    ax = fig.add_subplot(10,10,i+1)
    ax.imshow((image_data[i]).reshape(64,64),cmap='gray',interpolation="nearest")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fig.savefig("origin.png")



res_image = []
for row in eigvec[:5].dot(image_data.T).T:
    #res = eigvec[:5].dot(image_data.T).T
    res_image.append( (row*eigvec[:5].T).sum(axis=1).reshape(4096))


fig = plt.figure(figsize=(12,12))
for i in range(100):
    ax = fig.add_subplot(10,10,i+1)
    ax.imshow((res_image[i]+mean).reshape(64,64),cmap='gray',interpolation="nearest")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fig.savefig("reconstruct.png")
