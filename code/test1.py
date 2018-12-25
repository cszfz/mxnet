

import gluonbook as gb

from mxnet import gluon,init,nd
from mxnet.gluon import loss as gloss , nn,data as gdata
import numpy as np
import random

dirTrain='C:\\Users\\Raytine\\project\\image_train\\'

num_inputs = 7
num_examples = 51*51*51

def data_iter(batch_size,features,labels):
	num_examples=len(features)
	indices=list(range(num_examples))
	random,shuffle(indices)
	for i in range(0,num_examples,batch_size):
		j=nd.array(indices[i:min(i+batch_size,num_examples)])
		yield features.take(j),labels.take(j)

f=np.loadtxt(dirTrain+"image_train_moments.txt",delimiter=' ')
l=np.loadtxt(dirTrain+"image_train_angles.txt",delimiter=' ')
features=nd.array(f)
labels=nd.array(l)

net = nn.Sequential()
net.add(nn.Dense(10,activation='relu'),nn.Dense(3))
net.initialize(init.Normal(sigma=0.01))

batch_size=51*51

train_iter,test_iter=data_iter(batch_size,features,labels)
loss=gloss.L2Loss()

trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})




