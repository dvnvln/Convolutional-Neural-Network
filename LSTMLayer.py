import numpy as np
import math

from DenseLayer import sigmoid

class LSTMLayer:
    def __init__(self, neurons = 1, input_shape = (1,1), output_neurons = 1) -> None:
        self.uf = np.random.rand(neurons,input_shape[1])
        self.ui = np.random.rand(neurons,input_shape[1])
        self.uc = np.random.rand(neurons,input_shape[1])
        self.uo = np.random.rand(neurons,input_shape[1])
        self.h = np.zeros((input_shape[0]+1,neurons))
        self.c = np.zeros((input_shape[0]+1,neurons))
        self.wf = np.random.rand(neurons)
        self.wi = np.random.rand(neurons)
        self.wc = np.random.rand(neurons)
        self.wo = np.random.rand(neurons)
        self.bf = np.random.rand(neurons)
        self.bi = np.random.rand(neurons)
        self.bc = np.random.rand(neurons)
        self.bo = np.random.rand(neurons)

    def forward(self, inp):
        sig = np.vectorize(self.sigmoid)
        tanh = np.vectorize(self.tanh)
        for i in range(inp.shape[0]):
            ft = sig(np.matmul(self.uf,inp[i]) + np.multiply(self.wf,self.h[i]) + self.bf)
            it = sig(np.matmul(self.ui,inp[i]) + np.multiply(self.wi,self.h[i]) + self.bi)
            ct = tanh(np.matmul(self.uc,inp[i]) + np.multiply(self.wc,self.h[i]) + self.bc)
            # assign ct at the time
            self.c[i+1] = ct
            ot = sig(np.matmul(self.uo,inp[i]) + np.multiply(self.wo,self.h[i]) + self.bo)
            ht = np.multiply(ot,tanh(np.multiply(ft,self.c[i]) + np.multiply(it,ct)))
            # assign ht at the time
            self.h[i+1] = ht
        return self.h[1:]

    def sigmoid(self, x):
        return 1 / (1+ math.exp(-x))

    def tanh(self, x):
        return math.tanh(x)

# x = np.array([[1,2],[0.5,3]])
# model = LSTMLayer(1,input_shape=(2,2))
# model.uf[0][0] = 0.7
# model.uf[0][1] = 0.45
# model.ui[0][0] = 0.95
# model.ui[0][1] = 0.8
# model.uc[0][0] = 0.45
# model.uc[0][1] = 0.25
# model.uo[0][0] = 0.6
# model.uo[0][1] = 0.4
# model.wf[0] = 0.1
# model.bf[0] = 0.15
# model.wi[0] = 0.8
# model.bi[0] = 0.65
# model.wc[0] = 0.15
# model.bc[0] = 0.2
# model.wo[0] = 0.25
# model.bo[0] = 0.1
# print(model.forward(x))