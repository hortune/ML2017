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
fig = plt.figure()
plt.imshow(mean.reshape(64,64),cmap='gray')
fig.savefig("mean.png")

fig = plt.figure(figsize=(12,12))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    #ans = np.zeros(4096)
    #for q,p in zip(eigvec[i],image_data):
    #    ans += q*p
    ax.imshow((eigvec[i]).reshape(64,64),cmap='gray',interpolation="nearest")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
#plt.show()
fig.savefig("solution.png")
