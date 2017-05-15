import matplotlib.pyplot as plt
import numpy as np

fig,(ax1,ax2) = plt.subplots(2)


ax1.set_title('Accuracy/Epoch')
train_accur = [float(x[:-1]) for x in open('train_accuracy').readlines()]
valid_accur = [float(x[:-1]) for x in open('valid_accuracy').readlines()]

ax1.plot(list(range(1,301)),train_accur,color="red",label="train",lw=2)
ax1.plot(list(range(1,301)),valid_accur,color="blue",label="validation",lw=2)
ax1.legend(loc='lower right')

ax2.set_title('Loss/Epoch')
train_accur = [float(x[:-1]) for x in open('train_loss').readlines()]
valid_accur = [float(x[:-1]) for x in open('valid_loss').readlines()]

ax2.plot(list(range(1,301)),train_accur,color="red",label="train",lw=2)
ax2.plot(list(range(1,301)),valid_accur,color="blue",label="validation",lw=2)
ax2.legend(loc='upper right')
fig.savefig('acccuracy_comp.png')
