from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation

from net import *

import cfg


def CenterNet():
    input = Input(shape=(None, None, 3), name="input_image")
    backbone = get_backbone(input, cfg.backbone['type'])
    feature = backbone(input)

    def _conv_head(input, n_filter, name):
        curr_channel = input.shape[-1]
        x = Conv2D(curr_channel, kernel_size=3, padding='same', use_bias=False, name=name + '_conv1')(input)
        x = BatchNormalization(name=name + "_bn1")(x)
        x = Activation('relu', name=name + "_relu1")(x)

        x = Conv2D(n_filter, kernel_size=1, padding='same', use_bias=False, name=name + '_conv2')(x)
        x = BatchNormalization(name=name + "_bn2")(x)
        x = Activation('relu', name=name + "_relu2")(x)
        return x

    output_heads = {}
    for k, v in cfg.heads.items():
        output_heads[k] = _conv_head(feature, n_filter=v, name=k)

    model = Model(input, output_heads)
