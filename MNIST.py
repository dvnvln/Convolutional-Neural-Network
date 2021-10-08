from matplotlib.cm import get_cmap
from convolution import *
from DenseLayer import *
from poolinglayer import *
from SeqModel import *
import numpy as np
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('input shape',train_X.shape)

from matplotlib import pyplot

# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=get_cmap('gray'))
# pyplot.show()

train_X.resize(6000, 1, 28, 28)

inp = InputLayer()

convo = ConvolutionalLayer((1,28,28),(0,0),2,(1,4,4),(1,1))
det = DetectorLayer()
pool = PoolLayer(2,1,'average')

dense = DenseLayer((3, 1152), sigmoid)
out = OutputLayer()

arrlay = [inp, convo, det, pool, dense, out]

seq = SeqModel(arrlay)

# y = seq.forward_propagate_many(train_X[:10])

# seq.summary()

seq.train(train_X, train_y, 3, 0.2)