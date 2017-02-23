import sys
from numpy import genfromtxt,matmul
matrix1 = genfromtxt(sys.argv[1],delimiter=',')
matrix2 = genfromtxt(sys.argv[2],delimiter=',')
for i in matmul(matrix1,matrix2):
    print int(i)
