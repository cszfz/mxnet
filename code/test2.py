

# activation={'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):

	def __init__(self, **kwargs):
		super(MLP, self).__init__(**kwargs)
		self.hidden = nn.Dense(100, activation='relu')
		self.output = nn.Dense(3)

	def forward(self, x):
		return self.output(self.hidden(x))


x=nd.random.uniform(shape=(2,20))

net=MLP()
net.initialize()
net(x)

class MySequential(nn.Block):
	def __init__(self, **kwargs):
		super(MySequential, self).__init__(**kwargs)

	def add(self, block):
		self._children[block.name] = block

	def forward(self, x):
		for block in self._children.values():
			x = block
		return x


net = MySequential()
net.add(nn.Dense(256, activation='relu'))

net.initialize()
net(x)






class FancyMLP(nn.Block):
	def __init__(self, **kwargs):
		super(FancyMLP, self).__init__(**kwargs)
		self.rand_weight = self.params.get_constant(
			'rand_weight', nd.random.uniform(shape=(20,20)))

		self.dense = nn.Dense(20,activation='relu')

	def forward(self, x):
		x = self.dense(x)
		x=nd.relu(nd.dot(x, self.rand_weight.data())+1)
		x=self.dense(x)

		while x.norm().asscalar() > 1:
			x /=2

		if x.norm().asscalar() < 0.8:
			x *=10

		return x.sum()

net=FancyMLP()

net.initialize()

net(x)































































