import tensorflow as tf
import numpy as np
import coloredlogs, logging
import tensorlayer as tl


logger = logging.getLogger("resnet architectures")
coloredlogs.install(level='DEBUG')

__all__ = [
    'Resnet',
]

W_init=tf.truncated_normal_initializer(stddev=0.01, seed=24)
b_init=tf.constant_initializer(value=0.0)

# TODO: manage the methods for blocks better later once they go well
def identity_block(input_tensor, kernel_size, filters, stage, block_id, data_format='channels_last', train_bn=True):
    """ the skip connection no problem with output dims
    ACTIVATE(ADD((Conv2D ---> BN ---> (RELU) ---> Conv2D ---> BN ---> (RELU) ---> Conv2D ---> BN) + X)) """

    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block_id + '_branch'
    bn_name_base = 'bn' + str(stage) + block_id + '_branch'

    shortcut = input_tensor

    x = tl.layers.Conv2d(input_tensor, filter1, (1, 1), padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2a')
    x = tl.layers.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2a')

    x = tl.layers.Conv2d(x, filter2, (kernel_size, kernel_size), padding='SAME', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2b')
    x = tl.layers.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2b')

    x = tl.layers.Conv2d(x, filter3, (1, 1), padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2c')
    x = tl.layers.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2c')

    x = tl.layers.ElementwiseLayer([x, shortcut], combine_fn=tf.add, act=tf.nn.relu, name="add")
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block_id, strides=(2, 2), data_format='channels_last', train_bn=True):
    """ skip connection used if problems with dims, has convolution layer at shortcut """

    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block_id + '_branch'
    bn_name_base = 'bn' + str(stage) + block_id + '_branch'

    x = tl.layers.Conv2d(input_tensor, filter1, (1, 1), strides=strides, padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2a')
    x = tl.layers.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2a')
    x = tl.layers.Conv2d(x, filter2, (kernel_size, kernel_size), padding='SAME', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2b')
    x = tl.layers.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2b')

    x = tl.layers.Conv2d(x, filter3, (1, 1), padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2c')
    x = tl.layers.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2c')

    # add shortcut path changes
    shortcut = tl.layers.Conv2d(input_tensor, filter3, (1, 1), strides=strides, padding='VALID', data_format=data_format,
                W_init=W_init, b_init=b_init, name=conv_name_base + '1')
    shortcut = tl.layers.BatchNormLayer(shortcut, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '1')

    x = tl.layers.ElementwiseLayer([x, shortcut], combine_fn=tf.add, act=tf.nn.relu, name="add")
    return x


class Resnet50:
    """ resnet50 implementation using tensorlayer """
    def __init__(self, input):

        self.net = self.build(input)
        self.outputs = self.net.outputs
        self.all_params = list(self.net.all_params)
        self.all_layers = list(self.net.all_layers)
        self.all_drop = dict(self.net.all_drop)
        self.print_layers = self.net.print_layers
        self.print_params = self.net.print_params

    def build(self, input, data_format='channels_last', pooling='avg', train_bn=True):
        # stage 1
        net = tl.layers.InputLayer(input, name="input_image")
        net = tl.layers.ZeroPad2d(net, padding=(3,3), name="input_zeropad2d")
        net = tl.layers.Conv2d(net, 64, (7, 7), strides=(2, 2), padding='VALID', data_format=data_format,
                        W_init=W_init, b_init=b_init, name='stage1_conv2d')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=train_bn, name='stage1_bn')
        net = tl.layers.MaxPool2d(net, (3, 3), strides=(2, 2), padding='SAME', name="stage1_maxpool2d")
        # stage 2
        net = conv_block(net, 3, [64, 64, 256], stage=2, block_id='a', strides=(1, 1), train_bn=train_bn)
        net = identity_block(net, 3, [64, 64, 256], stage=2, block_id='b', train_bn=train_bn)
        net = identity_block(net, 3, [64, 64, 256], stage=2, block_id='c', train_bn=train_bn)

        # stage 3
        net = conv_block(net, 3, [128, 128, 512], stage=3, block_id='a', train_bn=train_bn)
        net = identity_block(net, 3, [128, 128, 512], stage=3, block_id='b', train_bn=train_bn)
        net = identity_block(net, 3, [128, 128, 512], stage=3, block_id='c', train_bn=train_bn)
        net = identity_block(net, 3, [128, 128, 512], stage=3, block_id='d', train_bn=train_bn)
        # stage 4
        net = conv_block(net, 3, [256, 256, 1024], stage=4, block_id='a', train_bn=train_bn)
        # for resnet50 the identity_block count is 5
        for i in range(5):
            net = identity_block(net, 3, [256, 256, 1024], stage=4, block_id=chr(98 + i), train_bn=train_bn)
        # stage 5
        net = conv_block(net, 3, [512, 512, 2048], stage=5, block_id='a', train_bn=train_bn)
        net = identity_block(net, 3, [512, 512, 2048], stage=5, block_id='b', train_bn=train_bn)
        net = identity_block(net, 3, [512, 512, 2048], stage=5, block_id='c', train_bn=train_bn)

        if pooling == 'avg':
            net = tl.layers.GlobalMeanPool2d(net, data_format=data_format, name='avg_pool')
        elif pooling == 'max':
            net = tl.layers.GlobalMaxPool2d(net, data_format=data_format, name='max_pool')

        # output layer
        net = tl.layers.FlattenLayer(net)
        net = tl.layers.DenseLayer(net, n_units=1000, act=tf.nn.softmax, name='output')

        return net
