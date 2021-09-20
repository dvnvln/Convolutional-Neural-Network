import numpy as np

class SeqModel:
    def __init__(self, arrOfLayers=[]):
        self.layers = arrOfLayers
    
    def add(self, layer):
        self.layers.append(layer)
    
    def summary(self):
        for layer in self.layers:
            print(layer)
            # print(layer.output)
    
    def inp(self, inp):
        self.layers[0].output = inp
    
    def forward_propagate(self):
        for i in range(1,len(self.layers)):
            # print(self.layers[i-1].type())
            # print(self.layers[i-1].output)
            # print("===========")
            self.layers[i].forward(self.layers[i-1].output)
        return self.layers[-1].output

    def forward_propagate_many(self, inputs, toFile=None):
        arr_output = []
        for instance in inputs:
            self.inp(instance)
            out = self.forward_propagate()
            if toFile:
                with open(toFile,"w+") as f:
                    f.write(out)
            else:
                print(out)
                print()
            arr_output.append(out)
        return arr_output
        
    def back_propagate(self):
        pass

class Layer:
    def __init__(self):
        pass
    
    def forward(self, inp):
        pass
    
class InputLayer:
    def __init__(self, size=0, inp=[]):
        self.output = inp
    
    def __repr__(self):
        return ""

class OutputLayer:
    def __init__(self):
        self.output = []
    
    def forward(self, inp):
        self.output = inp
        return(inp)
    
    def __repr__(self):
        return ""