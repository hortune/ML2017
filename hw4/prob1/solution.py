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
eigval,eigvec = np.linalg.eigh(covariance_matrix)
