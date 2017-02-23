import sys
from numpy import genfromtxt,matmul
matrix1 = genfromtxt(sys.argv[1],delimiter=',')
matrix2 = genfromtxt(sys.argv[2],delimiter=',')
ls = matmul(matrix1,matrix2)
ls.sort()
with open("ans.txt","w+") as fd:
    for i in ls:
        print(int(i),file=fd)
