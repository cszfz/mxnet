import numpy as np


fiel_dir='C:\\Users\\Raytine\\mxnet\\code\\'

l_train=np.loadtxt(fiel_dir+"l_train.txt",delimiter=' ')
l_test=np.loadtxt(fiel_dir+"l_test.txt",delimiter=' ')

l=l_train-l_test

l=np.abs(l)

s=np.sum(l,axis=1)

s[s<=4.5]=1
s[s>4.5]=0

print(len(s))
print(np.sum(s))

print('acc_rate:',np.sum(s)/len(s))