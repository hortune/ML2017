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

def question1():
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

def question2():
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
        ax.imshow((image_data[i]+mean).reshape(64,64),cmap='gray',interpolation="nearest")
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


def rmse(res,origin):
    error = 0
    for i,j in zip(res,origin):
        error += (i-j).dot(i-j)
    return sqrt(error/(100*4096))/255

def question3():
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
