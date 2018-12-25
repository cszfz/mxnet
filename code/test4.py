

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




loss=gloss.L2Loss()

trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})

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


def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
	params=None,lr=None,trainer=None):
	for epoch in range(num_epochs):

		train_l_sum=0
		train_acc_sum=0
		for X,Y in train_iter:
			with autograd.record():
				Y_hat=net(X)
				l=loss(net(X),Y)
			l.backward()
			if trainer is None:
				gb.sgd(params,lr,batch_size)
			else:
				trainer.step(batch_size)

			train_l_sum+=l.mean().asscalar()
			train_acc_sum+=accuracy(Y_hat,Y)
		test_acc=evaluate_accuracy(test_iter,net)

		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
				% (epoch + 1, train_l_sum / len(train_iter),
				train_acc_sum / len(train_iter), test_acc))

num_epochs=500

gb.train_ch3(net,data_iter,test_iter)