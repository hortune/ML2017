import matplotlib.pyplot as plt
import numpy as np
"""
x = [(1,0),(0,1),(0,-1),(-1,0),(0,2),(0,-2),(-2,0)]
y = [-1,-1,-1,1,1,1,1]
def gen_z1(x1,x2):
    return 2*x2*x2 - 4*x1 + 1

def gen_z2(x1,x2):
    return x1*x1 - 2*x2 - 3

xpos = [(gen_z1(i[0],i[1]),gen_z2(i[0],i[1])) for (i,j) in zip(x,y) if j == 1 ]
xneg = [(gen_z1(i[0],i[1]),gen_z2(i[0],i[1])) for (i,j) in zip(x,y) if j == -1 ]
print (xpos)
"""

fig,(ax1,ax2) = plt.subplots(2)

#fig.suptitle("training and validation",fontsize=20)

ax1.set_title('Accuracy/Epoch')
train_accur = [float(x[:-1]) for x in open('train_accuracy').readlines()]
valid_accur = [float(x[:-1]) for x in open('valid_accuracy').readlines()]

ax1.plot(list(range(1,101)),train_accur,color="red",label="train",lw=2)
ax1.plot(list(range(1,101)),valid_accur,color="blue",label="validation",lw=2)
ax1.legend(loc='upper left')

ax2.set_title('Loss/Epoch')
train_accur = [float(x[:-1]) for x in open('train_loss').readlines()]
valid_accur = [float(x[:-1]) for x in open('valid_loss').readlines()]

ax2.plot(list(range(1,101)),train_accur,color="red",label="train",lw=2)
ax2.plot(list(range(1,101)),valid_accur,color="blue",label="validation",lw=2)
ax2.legend(loc='upper left')
fig.savefig('acccuracy_comp_dnn.png')
