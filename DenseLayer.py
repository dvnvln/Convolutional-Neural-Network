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
        self.inp = inp.flatten()
        out = np.dot(np.concatenate((self.weights, self.bias),axis=1), np.append(inp,1))
        self.outputba = out
        self.output = self.act(out)
        return self.output
    
    def resetDelta(self):
        self.deltaWeight = np.zeros((self.weights.shape))

    def update_weights(self, learning_rate):
        # Update weight formula = w  + momentum * w + learning_rate * errors * output
        # Update bias formula = bias + momentum * bias + learning_rate * errors
        for i in range(self.shape[0]):
            self.weights[i] = self.weights[i] - (self.weights[i] + (learning_rate * self.deltaWeight[i] * self.inp))

        self.bias = self.bias - ((momentum * self.bias) + (learning_rate * self.deltaW))
        self.bias = self.bias - (self.bias + (learning_rate * self.deltaWeight))

        self.resetDelta()
    
    def backward(self, prev_errors):
        derivative_values = np.array([])
        derivative_values = np.append(derivative_values, get_derivative(self.act, self.output))

        self.deltaW += np.multiply(derivative_values, prev_errors)
        dE = np.matmul(prev_errors, self.weights)

        return dE

def linear(arr):
    return [x for x in arr]

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

def get_derivative(act, arr):
    res = []
    for x in arr:
        if(act == 'linear'):
            res.append(x)
        elif(act == 'sigmoid'):
            res.append(x * (1-x))
        elif(act == 'reLU'):
            if x >= 0:
                res.append(1)
            else:
                res.append(0)
    return res