import tensorflow as tf
import numpy as np
import tensorlayer.layers as tl

W_init=tf.truncated_normal(stddev=0.01, seed=24)
b_init=tf.constant_initializer(value=0.0)

class Resnet50:
    """ resnet50 implementation using tensorlayer """
    def __init__(self, inputs, data_format= “channels_last”):
        # TODO: check for batchnorm application axis when dataformat is channels_first
        # Stage 1
        net = tl.InputLayer(inputs, name="input_image")
        net = tl.ZeroPad2d(net, padding=(3,3), name="input_zeropad2d")
        net = tl.Conv2d(net, 64, (7, 7), strides=(2, 2), padding='VALID', data_format=data_format,
                        W_init=W_init, b_init=b_init, name='stage1_conv2d')
        net = tl.BatchNormLayer(net, act=tf.nn.relu, is_train=True, name='stage1_bn')
        C1 = net = tl.MaxPool2d(net, (3, 3), strides=(2, 2), padding='SAME', name="stage1_maxpool2d")


    # TODO: manage the methods for blocks better later once they go well
    def identity_block(input_tensor, kernel_size, filters, stage, block_id, train_bn=True):
        """ the skip connection no problem with output dims
        ACTIVATE(ADD((Conv2D ---> BN ---> (RELU) ---> Conv2D ---> BN ---> (RELU) ---> Conv2D ---> BN) + X)) """

        filter1, filter2, filter3 = filters
        conv_name_base = 'res' + str(stage) + block_id + '_branch'
        bn_name_base = 'bn' + str(stage) + block_id + '_branch'

        shortcut = input_tensor

        x = tl.Conv2d(input_tensor, filter1, (1, 1), strides=(1, 1), padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2a')
        x = tl.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2a')

        x = tl.Conv2d(x, filter2, (kernel_size, kernel_size), strides=(1, 1), padding='SAME', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2b')
        x = tl.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2b')

        x = tl.Conv2d(x, filter1, (1, 1), strides=(1, 1), padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2c')
        x = tl.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2c')

        x = tl.ElementwiseLayer([x, shortcut], combine_fn=tf.add, act=tf.nn.relu, name="add")
        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block_id, strides=(2, 2), train_bn=True):
        """ skip connection used if problems with dims, has convolution layer at shortcut """

        filter1, filter2, filter3 = filters
        conv_name_base = 'res' + str(stage) + block_id + '_branch'
        bn_name_base = 'bn' + str(stage) + block_id + '_branch'

        x = tl.Conv2d(input_tensor, filter1, (1, 1), strides=(1, 1), padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2a')
        x = tl.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2a')

        x = tl.Conv2d(x, filter2, (kernel_size, kernel_size), strides=(1, 1), padding='SAME', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2b')
        x = tl.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2b')

        x = tl.Conv2d(x, filter1, (1, 1), strides=(1, 1), padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '2c')
        x = tl.BatchNormLayer(x, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '2c')

        # add shortcut path changes
        shortcut = tl.Conv2d(input_tensor, filter3, (1, 1), strides=strides, padding='VALID', data_format=data_format,
                    W_init=W_init, b_init=b_init, name=conv_name_base + '1')
        shortcut = tl.BatchNormLayer(shortcut, act=tf.nn.relu, is_train=train_bn, name=bn_name_base + '1')

        x = tl.ElementwiseLayer([x, shortcut], combine_fn=tf.add, act=tf.nn.relu, name="add")
        return x
