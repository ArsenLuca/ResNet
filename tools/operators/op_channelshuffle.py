'''

Channel Shuffle Operator using NDArray GPU API

reimplement from ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun

by Arsen Zhang

'''
import mxnet as mx
import numpy as np

class ShuffleChannel(mx.operator.CustomOp):
    """ Channel Shuffle Operator """

    def __init__(self, num_group):
        super(ShuffleChannel, self).__init__()
        self._num_group = int(num_group)

    def forward(self, is_train, req, in_data, out_data, aux):
        """ forward computation """

        input_data = in_data[0]
        input_shape = input_data.shape

        n = input_shape[1] / self._num_group

        reshaped_data = input_data.reshape((input_shape[0], self._num_group, n, input_shape[-2], input_shape[-1]))
        transposed_reshaped_data = mx.nd.transpose(reshaped_data, axes=(0, 2, 1, 3, 4))
        shuffled_data = transposed_reshaped_data.reshape(input_shape)

        self.assign(out_data[0], req[0], shuffled_data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """ backward propagation """
        input_data = in_data[0]
        input_shape = input_data.shape

        n = input_shape[1] / self._num_group

        top_grad_reshaped = out_grad[0].reshape((input_shape[0], n, self._num_group, input_shape[-2], input_shape[-1]))
        transposed_reshaped_grad = mx.nd.transpose(top_grad_reshaped, axes=(0, 2, 1, 3, 4))
        bottom_grad = transposed_reshaped_grad.reshape(input_shape)

        self.assign(in_grad[0], req[0], bottom_grad)

@mx.operator.register("shufflechannel")
class ShuffleChannelProp(mx.operator.CustomOpProp):
    def __init__(self, num_group):
        super(ShuffleChannelProp, self).__init__(need_top_grad=True)
        self._num_group = num_group

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        input_shape = in_shape[0]
        output_shape = in_shape[0]

        return [input_shape], [output_shape], []
    
    def infer_type(self, in_type):
        dtype = in_type[0]

        return [dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return ShuffleChannel(self._num_group)
