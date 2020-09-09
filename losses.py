import tensorflow as tf


def focal_loss(hm_pred, hm_gt):
    alpha = 2.
    beta = 4.

    pos_mask = tf.cast(tf.equal(hm_gt, 1.), dtype=tf.float32)
    neg_mask = tf.cast(tf.less(hm_gt, 1.), dtype=tf.float32)

    pos_loss = -tf.math.log(hm_pred) * tf.math.pow(1. - hm_pred, alpha) * pos_mask

    neg_loss = -tf.math.log(1 - hm_pred) * tf.math.pow(hm_pred, alpha) * tf.math.pow(1. - hm_gt, beta) * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    loss = tf.cond(tf.greater(num_pos, 0), lambda : (pos_loss + neg_loss) / num_pos, lambda : neg_loss)
    return loss


def reg_l1_loss(y_pred, y_gt, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_gt * mask - y_pred * mask))
    loss = total_loss / (tf.reduce_sum(mask) + 1e-5)
    return loss

