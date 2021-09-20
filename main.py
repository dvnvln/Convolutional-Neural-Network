from convolution import *
from DenseLayer import *
from poolinglayer import *
from SeqModel import *
import numpy as np

inp = InputLayer()

convo = ConvolutionalLayer((3,3,3),(0,0),2,(3,2,2),(1,1))
filter1 = [[[0,-1],
            [1,0]],

            [[5,4],
            [3,2]],

            [[16,24],
            [68, -2]]]
filter2 = [[[60,22],
            [32,18]],

            [[35,46],
            [7,23]],
            
            [[78,81],
            [20, 42]]]
filters = np.array([filter1, filter2])
# convo.set_filter(filters)
det = DetectorLayer()
pool = PoolLayer(2,1,'average')

dense = DenseLayer((3,2), sigmoid)
out = OutputLayer()

arrlay = [inp, convo, det, pool, dense, out]

seq = SeqModel(arrlay)

input_array = [[[16,24, 32],
            [47, 18,26],
            [68, 12, 9]],

            [[26, 57, 43],
            [24, 21, 12],
            [2, 11, 19]],
            
            [[18, 47, 21],
            [4, 6, 12],
            [81, 22, 13]]]

seq.inp(np.array(input_array))
print(seq.forward_propagate())


seq.summary()