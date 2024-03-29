import os

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend
import tensorflow.keras.utils as keras_utils

BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101': ('34fb605428fcc7aa4d62f44404c11509',
                   '0f678c91647380debd923963594981b3')
}


def block0(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """
    A basic residual block.
    :param x: input tensor
    :param filters: integer, filters of the layer
    :param kernel_size: default 3, kernel size of the layer
    :param stride: default 1, stride of the first layer;
    :param conv_shortcut: default False, use convolution shortcut if True, otherwise identity shortcut.
    :param name: string, block label.
    :return: Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(filters, 1, stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, stride, padding='same', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, 1, padding='same', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])

    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """
    A residual block
    :param x: input tensor
    :param filters: int, filters of the bottleneck layer
    :param kernel_size: default 3, kernel size of the bottleneck layer
    :param stride: default 1, stride of the first layer.
    :param conv_shortcut: default True, use convolution shortcut if True, otherwise identity shortcut.
    :param name: string, block label
    :return: output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)

    else:
        shortcut = x

    x = layers.Conv2D(filters=filters, kernel_size=1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation(activation='relu', name=name + "_1_relu")(x)

    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', name=name + "_2_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation(activation='relu', name=name + "_2_relu")(x)

    x = layers.Conv2D(filters=4 * filters, kernel_size=1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation(activation='relu', name=name + "_out")(x)
    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """
    A residual block
    :param x: input tensor
    :param filters: int, filters of the bottleneck layer
    :param kernel_size: default 3, kernel size of the bottleneck layer
    :param stride: default 1, stride of the first layer.
    :param conv_shortcut: default True, use convolution shortcut if True, otherwise identity shortcut.
    :param name: string, block label
    :return: output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    pre_act = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_pre_act_bn')(x)
    pre_act = layers.Activation(activation='relu', name=name + '_pre_act_relu')(pre_act)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(pre_act)
    else:
        shortcut = layers.MaxPooling2D(pool_size=1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, name=name + "_1_conv")(pre_act)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation(activation='relu', name=name + "_1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, use_bias=False, name=name + "_2_conv")(
        x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation(activation='relu', name=name + "_2_relu")(x)

    x = layers.Conv2D(filters=4 * filters, kernel_size=1, name=name + "_3_conv")(x)
    x = layers.Add(name=name + "_out")([shortcut, x])
    return x


def block3(x, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True, name=None):
    """
    A residual block
    :param x: input tensor
    :param filters: int, filters of the bottleneck layer
    :param kernel_size: default 3, kernel size of the bottleneck layer
    :param stride: default 1, stride of the first layer.
    :param groups: default 32, group size for grouped convolution.
    :param conv_shortcut: default True, use convolution shortcut if True, otherwise identity shortcut.
    :param name: string, block label
    :return: output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    if conv_shortcut is True:
        shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride, use_bias=False, name=name + "_0_conv")(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)

    else:
        shortcut = x

    x = layers.Conv2D(filters=filters, kernel_size=1, use_bias=False, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation(activation='relu', name=name + "_1_relu")(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c, use_bias=False, name=name + '_2_conv')(
        x)
    x_shape = backend.int_shape(x)[1:-1]
    x = layers.Reshape(x_shape + (groups, c, c))(x)
    output_shape = None
    x = layers.Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(c)]), output_shape=output_shape,
                      name=name + '_2_reduce')(x)
    x = layers.Reshape(x_shape + (filters,))(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation(activation='relu', name=name + "_2_relu")(x)

    x = layers.Conv2D((64 // groups) * filters, kernel_size=1, use_bias=False, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation(activation='relu', name=name + "_out")(x)
    return x


def stack0(x, filters, blocks, stride1=2, name=None):
    """
    A set of stacked basic residual blocks.
    :param x: input tensor
    :param filters: integer, filters of layer in a block
    :param blocks: integer, blocks in the stacked blocks
    :param stride1: default 2, stride of the first layer in the first block.
    :param name: string, stack label
    :return: Output tensor for the stacked blocks
    """
    x = block0(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block0(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """
    A set of stacked residual blocks.
    :param x: input tensor
    :param filters: interger, filters of the bottleneck layer in a block
    :param blocks: integer, blocks in the stacked blocks
    :param stride1: default 2, stride of the first layer in the first block.
    :param name: string, stack label
    :return: Output tensor for the stacked blocks
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """
    A set of stacked residual blocks.
    :param x: input tensor
    :param filters: interger, filters of the bottleneck layer in a block
    :param blocks: integer, blocks in the stacked blocks
    :param stride1: default 2, stride of the first layer in the first block.
    :param name: string, stack label
    :return: Output tensor for the stacked blocks
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """
    A set of stackd residual blocks.
    :param x: input tensor
    :param filters: interger, filters of the bottleneck layer in a block
    :param blocks: integer, blocks in the stacked blocks
    :param stride1: default 2, stride of the first layer in the first block.
    :param groups: default 32, group size fo grouped convolution
    :param name: string, stack label
    :return: Output tensor for the stacked blocks
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def ResNet(stack_fn, preact, use_bias, model_name='resnet', include_top=True, weights='imagenet', input_tensor=None,
           input_shape=None, pooling=None, classes=1000, **kwargs):
    """
    instantiates the ResNet, ResNetV2, and ResNeXt architecture

    Optionally loads weights pre-trained on ImageNet.
    :param stack_fn: a function that returns output tensor for the stacked residual blocks
    :param preact: whether to use pre-activation or not (True for ResNetV2, False for ResNet and ResNeXt)
    :param use_bias: whether to use biases for convolutional layers or not (True for ResNet and ResNetV2, False for ResNext).
    :param model_name: string, model name
    :param include_top: whether to include the fully-connected layer at the top of the network.
    :param weights: one for 'None' (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
    :param input_tensor: optional Keras tensor (i.e. output of 'layers.Input()') to be use as image input for the model.
    :param input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
    :param pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    :param classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    :param kwargs:
    :return: a keras model instance
    :raises
        ValueError: in case of invalid argument for 'weights', or invalid input shape

    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        # if not backend.is_keras_tensor(input_tensor):
        #     img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        # else:
            img_input = input_tensor
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPool2D(name='max_pool')(x)

    # Create Model
    model = models.Model(img_input, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNet18(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,
             **kwargs):
    def stack_fn(x):
        x = stack0(x, 64, 2, stride1=1, name='conv2')
        x = stack0(x, 128, 2, name='conv3')
        x = stack0(x, 256, 2, name='conv4')
        x = stack0(x, 512, 2, name='conv5')
        return x

    return ResNet(stack_fn, preact=False, use_bias=True, model_name='resnet18', include_top=include_top,
                  weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes,
                  **kwargs)


def ResNet34(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,
             **kwargs):
    def stack_fn(x):
        x = stack0(x, 64, 3, stride1=1, name='conv2')
        x = stack0(x, 128, 4, name='conv3')
        x = stack0(x, 256, 5, name='conv4')
        x = stack0(x, 512, 3, name='conv5')
        return x

    return ResNet(stack_fn, preact=False, use_bias=True, model_name='resnet34', include_top=include_top,
                  weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes,
                  **kwargs)


def ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,
             **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x

    return ResNet(stack_fn, preact=False, use_bias=True, model_name='resnet50', include_top=include_top,
                  weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes,
                  **kwargs)


def ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,
              **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 23, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x

    return ResNet(stack_fn, preact=False, use_bias=True, model_name='resnet101', include_top=include_top,
                  weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes,
                  **kwargs)


def ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000,
              **kwargs):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 8, name='conv3')
        x = stack1(x, 256, 36, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x

    return ResNet(stack_fn, preact=False, use_bias=True, model_name='resnet152', include_top=include_top,
                  weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes,
                  **kwargs)


setattr(ResNet18, '__doc__', ResNet.__doc__)
setattr(ResNet34, '__doc__', ResNet.__doc__)
setattr(ResNet50, '__doc__', ResNet.__doc__)
setattr(ResNet101, '__doc__', ResNet.__doc__)
setattr(ResNet152, '__doc__', ResNet.__doc__)


if __name__ == '__main__':
    from test_classifier import  train
    resnet = ResNet18(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
    resnet = ResNet34(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
    # resnet = ResNet50(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
    # resnet = ResNet101(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
    # resnet = ResNet152(include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
    train(resnet)
