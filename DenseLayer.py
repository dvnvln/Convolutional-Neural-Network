import numpy as np

class DenseLayer:
    def __init__(self, size, activation, weights=[], name="denseboi"):
#       size example : (3,2) = there are 3 nodes, in which each has 2 coefficient
        if not weights:
            self.weights = np.random.rand(size[0], size[1])*5
        else:
            self.weights = weights
        self.shape = size
        self.output = self.weights
        self.bias = np.random.rand(size[0],1)
        self.act = activation
        self.name = name
        
    def __repr__(self):
        return f"\nDense Layer : {self.name}\nOutput Shape : {self.shape[0]}\nNParams : {self.shape[0] * self.shape[1]}\n"
    
    def forward(self, inp):
        # input should be array of Y length, where (X,Y) is the shape of the layer
        inp = inp.flatten()
        out = np.dot(np.concatenate((self.weights, self.bias),axis=1), np.append(inp,1))
        self.outputba = out
        self.output = self.act(out)
        return self.output

def ReLU(arr):
    return [(x if x > 0 else 0) for x in arr]

def sigmoid(arr):
    res = []
    for x in arr:
        res.append(1/(1+np.exp(-x)))
    return res

def softmax(arr):
    res = []
    for x in arr:
        e_x = np.exp(x-np.max(x))
        res.append(e_x / e_x.sum(axis=0))
    return res