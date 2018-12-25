

import gluonbook as gb

from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss , nn,data as gdata
import numpy as np
import random

dirTrain='C:\\Users\\Raytine\\project\\image_train\\'

num_inputs = 7
num_examples = 51*51*51


f=np.loadtxt(dirTrain+"image_train_moments.txt",delimiter=' ')
l=np.loadtxt(dirTrain+"image_train_angles.txt",delimiter=' ')
features=nd.array(f)
labels=nd.array(l)

batch_size=51*51

dataset=gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)




net = nn.Sequential()
net.add(nn.Dense(10,activation='relu'),nn.Dense(3))
net.initialize(init.Normal(sigma=0.01))




loss=gloss.L1Loss()

trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.3})

def accuracy(Y_hat,Y):
	x=(abs(Y_hat[0]-Y[0])<0.13)
	y=(abs(Y_hat[1]-Y[1])<0.13)
	z=(abs(Y_hat[2]-Y[2])<0.13)

	if x:
		if y:
			if z:
				return 1
	return 0

def evaluate_accuracy(data_iter,net):
	acc=0
	for X,Y in data_iter:
		acc+=accuracy(net(X),Y)

	return acc/len(data_iter)


num_epochs=500
for epoch in range(1,num_epochs+1):
	for X,Y in data_iter:
		with autograd.record():
			l=loss(net(X),Y)
		l.backward()
		trainer.step(batch_size)
	l=loss(net(features),labels)
	print('epoch%d,loss:%f' % (epoch,l.mean().asnumpy()))