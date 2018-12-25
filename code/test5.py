
import gluonbook as gb
from mxnet import nd
from mxnet.gluon import loss as gloss

batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

print(train_iter)
print(test_iter)