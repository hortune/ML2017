import numpy as np
import sys
from PIL import Image,ImageChops
img1 = Image.open(sys.argv[1])
img2 = Image.open(sys.argv[2])
a = np.asarray(img1)
b = np.asarray(img2)
c = np.zeros((a.shape[0],a.shape[1],a.shape[2]))
for i in range(0,a.shape[0]):
    for j in range(0,a.shape[1]):
        if tuple(a[i][j])==tuple(b[i][j]):
            continue
        else:
            c[i][j]=b[i][j]
im = Image.fromarray(np.uint8(c))
im.save('ans.png')
