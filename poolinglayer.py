import numpy as np

class DetectorLayer:
    def __init__(self, name = "detectorlayer"):
        self.input = []
        self.name = name

    def __repr__(self):
        return f"\Detector layer : {self.name}\nOutput_Shape : {self.output.shape[0]}*{self.output.shape[1]}*{self.output.shape[2]}"
    
    def forward(self, input):
        #ReLU
        self.input = input
        # print("Det")
        # print(self.input)
        self.output = np.maximum(self.input,0)
        # print(self.output)
        return self.output


class PoolLayer:
    def __init__(self, size_filter, size_stride, mode, name="poollayer"):
        self.size_filter = size_filter
        self.size_stride = size_stride
        self.mode = mode #average or max
        self.name = name

    def __repr__(self):
        return f"\Pool layer : {self.name}\nOutput_Shape : {self.pooled_result.shape[0]}*{self.pooled_result.shape[1]}*{self.pooled_result.shape[2]}"

    def forward(self, inputs):
        self.input = inputs
        channel_size = inputs.shape[0]
        new_width = int((inputs.shape[1] - self.size_filter) / self.size_stride) + 1
        new_height = int((inputs.shape[2] - self.size_filter) / self.size_stride) + 1

        pooled_result = np.zeros([channel_size, new_width, new_height], dtype=np.double)

        for i in range(0, channel_size):
            for j in range(0, new_width):
                for k in range(0, new_height):
                    x = j*self.size_stride
                    y = k*self.size_stride
                    if (self.mode.lower() == 'average'):
                        pooled_result[i,j,k] = '%.3f' % np.average(inputs[i, x:(x+self.size_filter), y:(y+self.size_filter)])
                    elif (self.mode.lower() == 'max'):
                        pooled_result[i,j,k] = np.max(inputs[i, x:(x+self.size_filter), y:(y+self.size_filter)])
                    else:
                        pass
        self.pooled_result = pooled_result
        self.output = pooled_result
        return pooled_result