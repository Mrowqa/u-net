import tensorflow as tf
from params import *


class UNet:
    def __init__(self, layers):
        self.x = tf.placeholder(tf.float32, [None, IMG_EDGE_SIZE, IMG_EDGE_SIZE, LABEL_CHANNELS])
        self.y_target = tf.placeholder(tf.float32, [None, IMG_EDGE_SIZE, IMG_EDGE_SIZE, CATEGORIES_CNT])

        signal = self.x
        for fno, frame_desc in enumerate(layers):
            with tf.variable_scope('Frame_{}'.format(fno)):
                for lno, l in enumerate(frame_desc):
                    signal = l.route_signal(signal, 'W{}_{}'.format(fno, lno))

        signal = Conv2D(3, CATEGORIES_CNT, False).route_signal(signal, 'W_out')
        # todo apply mask?
        signal = tf.nn.softmax(signal - self.y_target)
        # loss with softmax_xentropy with logits?
        # apply var that maps to one-hot-encoding output // %%

        # todo train step

    def train(self):
        pass


class Layer:
    pass


class Conv2D(Layer):
    def __init__(self, patch_edge, channels_out, add_relu=True):
        self.patch_edge = patch_edge
        self.channels_out = channels_out
        self.add_relu = add_relu

    def route_signal(self, signal, var_name):
        channels_in = signal.shape[3]
        weights = weight_variable(var_name, [self.patch_edge, self.patch_edge, channels_in, self.channels_out])
        signal = tf.nn.conv2d(signal, weights, strides=[1, 1, 1, 1], padding="SAME")
        if self.add_relu:
            signal = tf.nn.relu(signal)
        return signal


class Deconv2D(Layer):
    def __init__(self, patch_edge, channels_out):
        self.patch_edge = patch_edge
        self.channels_out = channels_out

    def route_signal(self, signal, _):
        signal = tf.layers.conv2d_transpose(signal, filters=self.channels_out, kernel_size=self.patch_edge,
                                            strides=self.patch_edge, padding="SAME")
        return signal


class MaxPool(Layer):
    def __init__(self, patch_edge):
        self.patch_edge = patch_edge

    def route_signal(self, signal, _):
        kernel_strides = [1, self.patch_edge, self.patch_edge, 1]
        signal = tf.nn.max_pool(signal, ksize=kernel_strides, strides=kernel_strides, padding='SAME')
        return signal


class Push(Layer):
    def __init__(self, stack):
        self.stack = stack

    def route_signal(self, signal, _):
        self.stack.append(signal)
        return signal


class Concat(Layer):
    def __init__(self, stack):
        self.stack = stack

    def route_signal(self, signal, _):
        prev_signal = self.stack.pop()
        signal = tf.concat([signal, prev_signal], axis=3)
        return signal


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())  # Xavier Glorot


def main():
    stack = []

    def double_conv(channels):
        return [
            Conv2D(3, channels),
            Conv2D(3, channels),
        ]

    def block_down(channels):
        return double_conv(channels) + [
            Push(stack),
            MaxPool(2),
        ]

    def block_up(channels):
        return [
            Deconv2D(2, channels),
            Concat(stack),
        ] + double_conv(channels)

    layers = [  # TODO note: check what are intermediate values
        block_down(16),
        block_down(32),
        block_down(64),
        block_down(128),
        double_conv(256),
        block_up(128),
        block_up(64),
        block_up(32),
        block_up(16),
    ]

    # TODO create scoped unet, for no weights clash?
    unet = UNet(layers)
