from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation

import losses
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
    return model


def compute_loss(pred, gt):
    hm_loss = losses.focal_loss(pred['hm'], gt['hm'])
    wh_loss = losses.reg_l1_loss(pred['wh'], gt['wh'], gt['ind'], gt['mask'])
    offset_loss = losses.reg_l1_loss(pred['offset'], gt['offset'], gt['ind'], gt['mask'])

    hm_kp_loss = losses.focal_loss(pred['hm_kp'], gt['hm_kp'])
    kp_offset_loss = losses.reg_l1_loss(pred['kp_offset'], gt['kp_offset'], gt['kp_ind'], gt['kp_mask'])
    kps_loss = losses.reg_l1_loss(pred['kps'], gt['kps'], gt['kp_ind'], gt['kp_mask'])

    total_loss = cfg.weight_loss['hm'] * hm_loss + cfg.weight_loss['wh'] * wh_loss + cfg.weight_loss[
        'offset'] * offset_loss + cfg.weight_loss['hm_kp'] * hm_kp_loss + cfg.weight_loss[
                     'kp_offset'] * kp_offset_loss + cfg.weight_loss['kps'] * kps_loss

    return {
        "total": total_loss,
        "hm": hm_loss,
        "offset": offset_loss,
        "wh": wh_loss,
        "hm_kp": hm_kp_loss,
        "kp_offset": kp_offset_loss,
        "kps": kps_loss
    }
