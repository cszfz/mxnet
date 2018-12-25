

import gluonbook as gb

from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss , nn,data as gdata
import numpy as np
import random

dirTrain='C:\\Users\\Raytine\\project\\image_train\\'




def accuracy(y_hat,y):
	r=y_hat-y
	result=r.asnumpy()
	result=np.abs(result)
	l=sum(i <= 0.5 for i in result)
	
	return l

def evaluate_accuracy(data_iter,net):
	acc=0
	for X,y in data_iter:
		acc+=accuracy(net(X),y)

	return acc/len(data_iter)


def train_ch3(net,train_iter,loss,num_epochs,batch_size,
	params=None,lr=None,trainer=None):
	for epoch in range(num_epochs):

		train_l_sum=0
		train_acc_sum=0
		for X,y in train_iter:
			with autograd.record():
				y_hat=net(X)
				l=loss(net(X),y)
				
			l.backward()
			if trainer is None:
				gb.sgd(params,lr,batch_size)
			else:
				trainer.step(batch_size)

			train_l_sum+=l.mean().asscalar()
			train_acc_sum+=accuracy(y_hat,y)

		print('epoch %d, loss %.4f, train acc %.3f'
			% (epoch + 1, train_l_sum / len(train_iter),
				train_acc_sum / (len(train_iter)*batch_size)))
		



num_inputs = 7
num_examples = 51*51*51


f=np.loadtxt(dirTrain+"image_train_moments.txt",delimiter=' ',ndmin=2)
l=np.loadtxt(dirTrain+"image_train_angles_z.txt",delimiter=' ',ndmin=2)
features=nd.array(f)
print(features)
labels=nd.array(l)
print(labels)
batch_size=51

trainset=gdata.ArrayDataset(features,labels)
train_iter = gdata.DataLoader(trainset,batch_size,shuffle=True)



net = nn.Sequential()
net.add(nn.Dense(100,activation='sigmoid'),nn.Dense(100,activation='sigmoid'),nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))




loss=gloss.L2Loss()

trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.01})



num_epochs=5

train_ch3(net,train_iter,loss,num_epochs,batch_size,None,None,trainer)