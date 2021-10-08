import numpy as np
from poolinglayer import DetectorLayer, PoolLayer

class ConvolutionalLayer:
    # size input dan size filter menggunakan format (jumlah_channel, tinggi, lebar)
    # size padding dan stride meunggunakan format (tinggi, lebar)
    # def __init__(self, size_input: tuple[int, int, int], padding: tuple[int, int], n_filter: int, size_filter: tuple[int, int, int], size_stride: tuple[int, int]):
    #     if(size_input[0] != size_filter[0]):
    #         raise ValueError('Jumlah channel input dan filter tidak sesuai!!')
    #     self.size_input = size_input
    #     self.padding = padding
    #     self.n_filter = n_filter
    #     self.size_filter = size_filter
    #     self.shape_filter = (n_filter,) + size_filter
    #     self.size_stride = size_stride
    #     self.filters = np.zeros(self.shape_filter)
    #     self.bias = np.zeros(n_filter)
    #     self.detector = 'ReLU'
    def __init__(self, size_input, padding, n_filter, size_filter, size_stride, name="Convolution_default_name"):
        if(size_input[0] != size_filter[0]):
            raise ValueError('Jumlah channel input dan filter tidak sesuai!!')
        self.size_input = size_input
        self.padding = padding
        self.n_filter = n_filter
        self.size_filter = size_filter
        self.shape_filter = (n_filter,) + size_filter
        self.size_stride = size_stride
        self.filters = np.random.rand(*self.shape_filter)
        self.bias = np.zeros(n_filter)
        self.detector = 'ReLU'
        self.name = name

    def __repr__(self):
        return f"\nConvolutional layer : {self.name}\nOutput_Shape : {self.output.shape[0]}*{self.output.shape[1]}*{self.output.shape[2]}\nNParams : {(self.size_filter[0]*self.size_filter[2]*self.size_filter[1]+1) * self.n_filter}\n"


    def set_detector(self, detector: str = 'ReLU'):
        if(detector == 'ReLU' or detector == 'sigmoid'):
            self.detector = detector
        else:
            raise ValueError('Detector yang tersedia hanya ReLU dan sigmoid.')

    def set_filter(self, filters: np.ndarray):
        if filters.shape != self.shape_filter:
            raise ValueError('Shape filter tidak sesuai.')
        self.filters = filters

    def calculate_receptive_map_size(self):
        return (self.size_filter[0],int((self.size_input[1]-self.size_filter[1]+self.padding[0]+self.size_stride[0])/self.size_stride[0]), int((self.size_input[2]-self.size_filter[2]+self.padding[1]+self.size_stride[1])/self.size_stride[1]))

    def forward(self, input: np.ndarray):
        if(input.shape != self.size_input):
            raise ValueError('Input tidak sesuai!!')
        shape_receptive_field = self.calculate_receptive_map_size()
        output = np.zeros((self.n_filter, shape_receptive_field[1], shape_receptive_field[2]))
        # print(output)
        self.input = input
        for filter in range(self.n_filter):
            for channel in range(shape_receptive_field[0]):
                # print(output)
                for t in range(shape_receptive_field[1]):
                    for l in range(shape_receptive_field[2]):
                        rec_start_t = max(0,t*self.size_stride[0]-self.padding[0])
                        rec_end_t = min(input.shape[1],t*self.size_stride[0]-self.padding[0]+self.size_filter[1])
                        rec_start_l = max(0,l*self.size_stride[1]-self.padding[1])
                        rec_end_l = min(input.shape[2],l*self.size_stride[1]-self.padding[1]+self.size_filter[2])
                        output[filter, t, l] += np.sum(np.multiply(input[channel,rec_start_t:rec_end_t,rec_start_l:rec_end_l],self.filters[filter, channel])) + self.bias[filter]
        self.output = output
        return output

    def backward(self, prev_errors):
        self.derr = prev_errors * self.input
        return self.derr

    def update_weights(self, learning_rate):
        self.filters -= learning_rate*self.derr

# conv = ConvolutionalLayer((3,3,3),(0,0),2,(3,2,2),(1,1))
# filter1 = [[[0,-1],
#             [1,0]],

#             [[5,4],
#             [3,2]],

#             [[16,24],
#             [68, -2]]]
# filter2 = [[[60,22],
#             [32,18]],

#             [[35,46],
#             [7,23]],
            
#             [[78,81],
#             [20, 42]]]

# input_array = [[[16,24, 32],
#             [47, 18,26],
#             [68, 12, 9]],

#             [[26, 57, 43],
#             [24, 21, 12],
#             [2, 11, 19]],
            
#             [[18, 47, 21],
#             [4, 6, 12],
#             [81, 22, 13]]]

# filters = np.array([filter1, filter2])
# inputs = np.array(input_array)
# print(filters)

# conv.set_filter(filters)

# # print(conv.calculate_receptive_map_size())
# output = conv.conv(inputs)
# print("hasil convolution", output)

# det = DetectorLayer(output)
# hasilDet = det.forward()

# pool = PoolLayer(2,2,"max")
# hasilPool = pool.forward(hasilDet)
# print('hasil pooling',hasilPool)