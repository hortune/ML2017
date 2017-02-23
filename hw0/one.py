import sys
from numpy import genfromtxt,matmul
matrix1 = genfromtxt(sys.argv[1],delimiter=',')
matrix2 = genfromtxt(sys.argv[2],delimiter=',')
ls = matmul(matrix1,matrix2)
if type(ls[0]) == type(matrix1):
    ls = [j for i in ls for j in i]
ls.sort()
with open("ans_one.txt","w+") as fd:
    for i in ls:
        print(int(i),file=fd)
