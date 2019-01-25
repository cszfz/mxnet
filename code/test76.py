

import gluonbook as gb

from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss , nn,data as gdata
import mxnet as mx
import numpy as np
import random
import time

dirTrain='D:\\image\\txt\\2l\\'

ctx=mx.gpu()

f=np.loadtxt(dirTrain+"image_train_features.txt",delimiter=' ')
l=np.loadtxt(dirTrain+"image_train_labels.txt",delimiter=' ')


features=nd.array(f).copyto(ctx)
labels=nd.array(l).copyto(ctx)
labels_test=nd.zeros(labels.shape,ctx)

data_num=len(f)
batch_size=500

dataset=gdata.ArrayDataset(features,labels)
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)




net = nn.Sequential()
net.add(nn.Dense(128,activation='relu'),
	nn.BatchNorm(),
	nn.Dense(128,activation='relu'),
	nn.BatchNorm(),
	nn.Dense(3))
net.initialize(init.Xavier(),ctx=ctx)




loss=gloss.L2Loss()

trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.002, 'momentum': 0.9})


def accuracy(y_hat,y,error):
	sum_acc=0
	yy=y_hat-y
	yyy=yy.asnumpy()
	yyy=np.abs(yyy)
	i=0
	for i,val in enumerate(yyy):
		sum_acc=sum_acc+equal(val,error)

	return sum_acc

def equal(y,error):
	error=0.5
	xx=(y[0]<error)
	yy=(y[1]<error)
	zz=(y[2]<error)

	if xx:
		if yy:
			if zz:
				return 1
	return 0

def accuracy_sum(y_hat,y,error):
	sum_acc=0
	yy=y_hat-y
	yyy=yy.asnumpy()
	yyy=np.abs(yyy)
	i=0
	for i,val in enumerate(yyy):
		sum_acc=sum_acc+equal_sum(val,error)

	return sum_acc

def equal_sum(y,error):
	sum_acc=y[0]+y[1]+y[2]

	if sum_acc<error:
		return 1

	return 0



def train_ch3(net,train_iter,loss,num_epochs,batch_size,
	params=None,lr=None,trainer=None):
	

	for epoch in range(num_epochs):

		if epoch%100==0:
			trainer.set_learning_rate(trainer.learning_rate*0.5)
			print(trainer.learning_rate)


		train_acc_sum=0
		train_acc_sum1=0
		train_acc_sum2=0
		train_acc_sum3=0

		flag=0

		train_l_sum=0
		train_acc_sum=0
		for X,y in train_iter:
			with autograd.record():
				y_hat=net(X)
				l=loss(y_hat,y)
				
			l.backward()
			if trainer is None:
				gb.sgd(params,lr,batch_size)
			else:
				trainer.step(batch_size)

			train_l_sum+=l.mean().asscalar()
			if epoch==(num_epochs-1):
				labels[flag:flag+len(y)]=y
				labels_test[flag:flag+len(y)]=y_hat
				flag=flag+len(y)

			train_acc_sum+=accuracy(y_hat,y,0.51)
			train_acc_sum1+=accuracy_sum(y_hat,y,1.5)
			train_acc_sum2+=accuracy_sum(y_hat,y,3.0)
			train_acc_sum3+=accuracy_sum(y_hat,y,4.5)


		
			
		print('epoch %d, loss %.4f, train_acc %f, train_acc1 %f, train_acc2 %f, train_acc3 %f'
				% (epoch + 1, train_l_sum / len(train_iter),
					train_acc_sum/data_num, train_acc_sum1/data_num, 
					train_acc_sum2/data_num, train_acc_sum3/data_num))
num_epochs=10001



train_ch3(net,data_iter,loss,num_epochs,batch_size,None,None,trainer)

np.savetxt((time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))+'l_train.txt',labels.asnumpy(),fmt='%f')
np.savetxt((time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))+'l_test.txt',labels_test.asnumpy(),fmt='%f')