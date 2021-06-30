import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2
from yolov3.utils import read_class_names
from yolov3.configs import *
import tensorflow.keras.backend as K
from  yolov3.mobilenetv3 import YOLOV4_MobileNet

STRIDES         = np.array(YOLO_STRIDES)
ANCHORS         = (np.array(YOLO_ANCHORS_LS).T/STRIDES).T


class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
# 定义bn层


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = LeakyReLU(alpha=0.1)(conv)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)
    residual_output = short_cut + conv
    return residual_output


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def darknet53(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)
    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)
    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)
    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)
    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)
    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)
    return route_1, route_2, input_data


def cspdarknet53(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = convolutional(input_data, (1, 1, 2048, 512))

    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    return route_1, route_2, input_data


def cspdarknet53s(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 64, 64), bn=False,activate=False)
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    for i in range(1):
        input_data = residual_block(input_data, 64, 32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), bn=False,activate=False)

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = BatchNormalization()(input_data)
    input_data = mish(input_data)
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 128, 64),bn=False,activate=False)
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(3):
        input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64),bn=False,activate=False)
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = BatchNormalization()(input_data)
    input_data = mish(input_data)
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 256, 128),bn=False,activate=False)
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(15):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), bn=False,activate=False)
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = BatchNormalization()(input_data)
    input_data = mish(input_data)
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), bn=False,activate=False)
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(15):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), bn=False,activate=False)
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = BatchNormalization()(input_data)
    input_data = mish(input_data)
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), bn=False,activate=False)
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(7):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 512),bn=False,activate=False)
    input_data = tf.concat([input_data, route], axis=-1)
    input_data = BatchNormalization()(input_data)
    input_data = mish(input_data)
    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 1024))
    rout__1 = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 512))
    input_data = convolutional(input_data, (1, 1, 512, 512))
    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 512))
    input_data = tf.concat([rout__1,input_data],axis=-1)
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    return route_1, route_2, input_data


def darknet19_tiny(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 16))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 16, 32))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 32, 64))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 64, 128))
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 128, 256))
    route_1 = input_data
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 256, 512))
    input_data = MaxPool2D(2, 1, 'same')(input_data)
    input_data = convolutional(input_data, (3, 3, 512, 1024))

    return route_1, input_data


def cspdarknet53_tiny(input_data): # not sure how this should be called
    input_data = convolutional(input_data, (3, 3, 3, 32), downsample=True)
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)
    input_data = convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 256))
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 512, 512))

    return route_1, input_data


def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = darknet53(input_layer)
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 185)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1,  512,  256))
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 185)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)
    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS + 185)), activate=False, bn=False)
    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = cspdarknet53(input_layer)
    route = conv
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)
    route_2 = convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    route_1 = convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5 + 180)), activate=False, bn=False)

    conv = convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5 + 180)), activate=False, bn=False)

    conv = convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5 + 180)), activate=False, bn=False)
    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv4_Large(input_layer, NUM_CLASS):
    route_1, route_2, conv = cspdarknet53s(input_layer)
    route = conv
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)
    route_2 = convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    route_1 = convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5 + 180)), activate=False, bn=False)

    conv = convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5 + 180)), activate=False, bn=False)

    conv = convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5 + 180)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv3_tiny(input_layer, NUM_CLASS):
    route_1, conv = darknet19_tiny(input_layer)

    conv = convolutional(conv, (1, 1, 1024, 256))
    conv_lobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    
    conv = tf.concat([conv, route_1], axis=-1)
    conv_mobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]


def YOLOv4_tiny(input_layer, NUM_CLASS):
    route_1, conv = cspdarknet53_tiny(input_layer)

    conv = convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]


def Create_Yolo(input_size=416, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = len(read_class_names(CLASSES))
    input_layer  = Input([input_size, input_size, channels])
    # 设置网络的输入大小，默认[416,416,3]
    print('网络的输入大小为:{}'.format(input_size))

    if TRAIN_YOLO_TINY:
        if YOLO_TYPE == "yolov4":
            conv_tensors = YOLOv4_tiny(input_layer, NUM_CLASS)
        if YOLO_TYPE == "yolov3":
            conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if YOLO_TYPE == "yolov4_large":
            conv_tensors = YOLOv4_Large(input_layer, NUM_CLASS)
        if YOLO_TYPE == "yolov4":
            conv_tensors = YOLOv4(input_layer, NUM_CLASS)
        if YOLO_TYPE == "yolov3":
            conv_tensors = YOLOv3(input_layer, NUM_CLASS)
        if YOLO_TYPE == 'yolov4-mobilenet':
            conv_tensors = YOLOV4_MobileNet(input_layer, NUM_CLASS)
            print('模型的主干网络为:MobilNet V3')

    output_tensors = []

    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    Yolo = tf.keras.Model(input_layer, output_tensors)
    # Yolo.summary()
    return Yolo


def decode(conv_output, NUM_CLASS, i=0):
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 185 + NUM_CLASS))
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf,conv_raw_angel, conv_raw_prob = tf.split(conv_output, (2, 2, 1, 180, NUM_CLASS), axis=-1)
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size, dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = (2.0*tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_angel=tf.sigmoid(conv_raw_angel)
    pred_prob = tf.sigmoid(conv_raw_prob)
    return tf.concat([pred_xywh, pred_conf, pred_angel, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    enclose_area = enclose[..., 0] * enclose[..., 1]

    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


# testing (should be better than giou)
def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / (boxes2[..., 3]+0.000001)
    ar_pred = boxes1[..., 2] / (boxes1[..., 3]++ 0.000001)

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred)+ 0.000001)
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term
# ciou与giou都不是很稳定,训练环节有时候会导致nan损失,对于不稳定的因素可能是由于wh的预测范围是对用于exp(0,+inf),所以预测的wh范围是(0,inf)
# 所以会导致训练环节出现不稳定的因素导致损失为nan
# 当然也可以继续使用可代替的办法如iou损失以及mse均方差损失代替为nan的情况


def rboxes2corner(boxes):
    x, y, w, h, angel = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4], boxes[..., 4:5]

    cos_w = 0.5 * w * tf.math.cos(angel * 3.14159 / 180)
    sin_w = 0.5 * w * tf.math.sin(angel * 3.14159 / 180)
    cos_h = 0.5 * h * tf.math.cos(angel * 3.14159 / 180)
    sin_h = 0.5 * h * tf.math.sin(angel * 3.14159 / 180)

    x0 = x - cos_w - sin_h
    y0 = y + sin_w - cos_h

    x1 = x - cos_w + sin_h
    y1 = y + sin_w + cos_h

    x2 = x + cos_w + sin_h
    y2 = y - sin_w + cos_h

    x3 = x + cos_w - sin_h
    y3 = y - sin_w - cos_h

    cur_corner = tf.concat([x0, y0, x1, y1, x2, y2, x3, y3], axis=-1)
    return cur_corner


'''
def bbox_riou(pred_bbox,gt_bbox):
    pred_corner = rboxes2corner(pred_bbox)
    gt_corner = rboxes2corner(gt_bbox)
    return None
'''

def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=YOLO_COCO_CLASSES):
    bboxes = bboxes[... , :4]

    NUM_CLASS = len(read_class_names(CLASSES))
    conv_shape  = tf.shape(conv)
    # 获取未解码之前的输出的形状信息
    # [batch_size,52,52,3*(80+5+180)]
    # [batch_size,26,26,3*(80+5+180)]
    # [batch_size,13,13,3*(80+5+180)]

    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + 180 + NUM_CLASS))
    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_angel=conv[:, :, :, :,5:(5+180)]
    conv_raw_prob = conv[:, :, :, :, (5+180):]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_angel = label[:,:,:,:,5:(5+180)]
    label_prob    = label[:, :, :, :, (5+180):]

    giou = tf.expand_dims(bbox_iou(pred_xywh, label_xywh), axis=-1)

    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)

    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    angel_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_angel,logits=conv_raw_angel)

    giou_loss = tf.reduce_mean(giou_loss, axis=-1)
    conf_loss = tf.reduce_mean(conf_loss, axis=-1)
    angel_loss = tf.reduce_mean(angel_loss, axis=-1)
    prob_loss = tf.reduce_mean(prob_loss, axis=-1)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3]))
    angel_loss = tf.reduce_mean(tf.reduce_sum(angel_loss,axis=[1,2,3]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3]))
    return giou_loss, conf_loss, angel_loss, prob_loss
