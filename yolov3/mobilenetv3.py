import tensorflow as tf
import numpy  as np
from tensorflow.keras import backend
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv2D, Dense, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Input, Multiply,
                                     MaxPooling2D,
                                     Concatenate,
                                     UpSampling2D,
                                     Reshape)
from tensorflow.keras import backend as K
from functools import wraps
from functools import reduce
from tensorflow.keras.models import Model


def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)


def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0


def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _bneck(inputs, expansion, alpha, out_ch, kernel_size, stride, se_ratio, activation,
                        block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    out_channels = _make_divisible(out_ch * alpha, 8)
    exp_size = _make_divisible(in_channels * expansion, 8)
    x = inputs
    prefix = 'expanded_conv/'
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(exp_size,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + 'expand/BatchNorm')(x)
        x = _activation(x, activation)

    x = DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = _activation(x, activation)

    if se_ratio:
        reduced_ch = _make_divisible(exp_size * se_ratio, 8)
        y = GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(x)
        y = Reshape([1, 1, exp_size], name=prefix + 'reshape')(y)
        y = Conv2D(reduced_ch,
                          kernel_size=1,
                          padding='same',
                          use_bias=True,
                          name=prefix + 'squeeze_excite/Conv')(y)
        y = Activation("relu", name=prefix + 'squeeze_excite/Relu')(y)
        y = Conv2D(exp_size,
                          kernel_size=1,
                          padding='same',
                          use_bias=True,
                          name=prefix + 'squeeze_excite/Conv_1')(y)
        x = Multiply(name=prefix + 'squeeze_excite/Mul')([Activation(hard_sigmoid)(y), x])

    x = Conv2D(out_channels,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                                  name=prefix + 'project/BatchNorm')(x)

    if in_channels == out_channels and stride == 1:
        x = Add(name=prefix + 'Add')([inputs, x])
    return x


def MobileNetV3(inputs, alpha=1.0, kernel=5, se_ratio=0.25):
    if alpha not in [0.75, 1.0]:
        raise ValueError('Unsupported alpha - `{}` in MobilenetV3, Use 0.75, 1.0.'.format(alpha))
    # 416,416,3 -> 208,208,16
    x = Conv2D(16, kernel_size=3, strides=(2, 2), padding='same',
               use_bias=False,
               name='Conv')(inputs)
    x = BatchNormalization(axis=-1,
                           epsilon=1e-3,
                           momentum=0.999,
                           name='Conv/BatchNorm')(x)
    x = Activation(hard_swish)(x)

    # 208,208,16 -> 208,208,16
    x = _bneck(x, 1, 16, alpha, 3, 1, None, 'relu', 0)

    # 208,208,16 -> 104,104,24
    x = _bneck(x, 4, 24, alpha, 3, 2, None, 'relu', 1)
    x = _bneck(x, 3, 24, alpha, 3, 1, None, 'relu', 2)

    # 104,104,24 -> 52,52,40
    x = _bneck(x, 3, 40, alpha, kernel, 2, se_ratio, 'relu', 3)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', 4)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', 5)
    feat1 = x

    # 52,52,40 -> 26,26,112
    x = _bneck(x, 6, 80, alpha, 3, 2, None, 'hardswish', 6)
    x = _bneck(x, 2.5, 80, alpha, 3, 1, None, 'hardswish', 7)
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 'hardswish', 8)
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 'hardswish', 9)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 'hardswish', 10)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 'hardswish', 11)
    feat2 = x

    # 26,26,112 -> 13,13,160
    x = _bneck(x, 6, 160, alpha, kernel, 2, se_ratio, 'hardswish', 12)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 'hardswish', 13)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 'hardswish', 14)
    feat3 = x

    return feat1, feat2, feat3


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def relu6(x):
    return K.relu(x, max_value=6)


# --------------------------------------------------#
#   单次卷积DarknetConv2D
#   正则化系数为5e-4
#   如果步长为2则自己设定padding方式。
# --------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + Relu6
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Activation(relu6))


# ---------------------------------------------------#
#   深度可分离卷积块
#   DepthwiseConv2D + BatchNormalization + Relu6
# ---------------------------------------------------#
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha=1,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False)(inputs)

    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1))(x)
    x = BatchNormalization()(x)
    return Activation(relu6)(x)


# ---------------------------------------------------#
#   进行五次卷积
# ---------------------------------------------------#
def make_five_convs(x, num_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = _depthwise_conv_block(x, num_filters * 2, alpha=1)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = _depthwise_conv_block(x, num_filters * 2, alpha=1)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x


def YOLOV4_MobileNet(input_layer, NUM_CLASS, alpha=1, num_anchors=3):
    feat1, feat2, feat3 = MobileNetV3(input_layer, alpha=alpha)

    P5 = DarknetConv2D_BN_Leaky(int(512 * alpha), (1, 1))(feat3)
    P5 = _depthwise_conv_block(P5, int(1024 * alpha))
    P5 = DarknetConv2D_BN_Leaky(int(512 * alpha), (1, 1))(P5)
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(int(512 * alpha), (1, 1))(P5)
    P5 = _depthwise_conv_block(P5, int(1024 * alpha))
    P5 = DarknetConv2D_BN_Leaky(int(512 * alpha), (1, 1))(P5)

    P5_upsample = compose(DarknetConv2D_BN_Leaky(int(256 * alpha), (1, 1)), UpSampling2D(2))(P5)

    P4 = DarknetConv2D_BN_Leaky(int(256 * alpha), (1, 1))(feat2)
    P4 = Concatenate()([P4, P5_upsample])
    P4 = make_five_convs(P4, int(256 * alpha))

    P4_upsample = compose(DarknetConv2D_BN_Leaky(int(128 * alpha), (1, 1)), UpSampling2D(2))(P4)

    P3 = DarknetConv2D_BN_Leaky(int(128 * alpha), (1, 1))(feat1)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = make_five_convs(P3, int(128 * alpha))

    # ---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    # ---------------------------------------------------#
    P3_output = _depthwise_conv_block(P3, int(256 * alpha))
    P3_output = DarknetConv2D(num_anchors * (NUM_CLASS + 5 + 180), (1, 1))(P3_output)

    P3_downsample = _depthwise_conv_block(P3, int(256 * alpha), strides=(2, 2))
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_five_convs(P4, int(256 * alpha))

    # ---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    # ---------------------------------------------------#
    P4_output = _depthwise_conv_block(P4, int(512 * alpha))
    P4_output = DarknetConv2D(num_anchors * (NUM_CLASS + 5 + 180), (1, 1))(P4_output)

    P4_downsample = _depthwise_conv_block(P4, int(512 * alpha), strides=(2, 2))
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_five_convs(P5, int(512 * alpha))

    # ---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    # ---------------------------------------------------#
    P5_output = _depthwise_conv_block(P5, int(1024 * alpha))
    P5_output = DarknetConv2D(num_anchors * (NUM_CLASS + 5 + 180), (1, 1))(P5_output)

    return [P3_output, P4_output, P5_output]