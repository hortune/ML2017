from random import randint


for i in range(0,10):
    with open(str(i)+"-1.in","w+",encoding='UTF-8') as fd, open(str(i)+"-2.in","w+",encoding="UTF-8") as fd1:
        x1 =randint(1,100)
        x2 =randint(1,100)
        x3 =randint(1,100)
        for q in range(0,x1):
            k=[str(randint(0,100)) for j in range(0,x2)]
            print(",".join(k),file=fd)
        for q in range(0,x2):
            k=[str(randint(0,100)) for j in range(0,x3)]
            print(",".join(k),file=fd1)

import sys
from numpy import genfromtxt,matmul

for i in range(0,10):
    matrix1 = genfromtxt(str(i)+"-1.in",delimiter=',')
    matrix2 = genfromtxt(str(i)+"-2.in",delimiter=',')
    ls = matmul(matrix1,matrix2)
    if type(ls[0]) == type(matrix1):
        ls = [j for i in ls for j in i]
    ls.sort()
    with open(str(i)+".out","w+") as fd:
        for i in ls:
            print(int(i),file=fd)
