from net.resnet_common import *
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose
import tensorflow as tf

backbone = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
}


def get_backbone(input, type):
    network = backbone[type]
    return network(include_top=False, weights="imagenet", input_tensor=input)


def upsampling(inputs, method='deconv'):
    assert method in ["resize", "deconv"]
    if method == "resize":
        input_shape = tf.shape(inputs)
        output = UpSampling2D()(inputs)
        # tf.image.resize(inputs, (input_shape[1] * 2, input_shape[2] * 2), method="nearest")
    if method == "deconv":
        num_filters = inputs.shape[-1]
        output = Conv2DTranspose(filters=num_filters, kernel_size=4, strides=2, padding='same')(
            inputs)
    return output
