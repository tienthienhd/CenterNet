import time

import tensorflow as tf
from tqdm import tqdm

from generator import *
from model import CenterNet, compute_loss

# tf.compat.v1.disable_v2_behavior()
# sess = tf.compat.v1.Session()
if cfg.debug:
    tf.debugging.experimental.enable_dump_debug_info("logs", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# tf.compat.v1.keras.backend.set_session(sess)


def train():
    # define dataset
    num_train_imgs = len(open(cfg.train_data_file, 'r').readlines())
    num_train_batch = int(math.ceil(float(num_train_imgs) / cfg.batch_size))
    num_test_imgs = len(open(cfg.test_data_file, 'r').readlines())
    num_test_batch = int(math.ceil(float(num_test_imgs) / 2))

    train_dataset = tf.data.TextLineDataset(cfg.train_data_file)
    train_dataset = train_dataset.shuffle(num_train_imgs)
    train_dataset = train_dataset.batch(cfg.batch_size)
    train_dataset = train_dataset.map(
        lambda x: tf.compat.v1.py_func(get_data, inp=[x], Tout=[tf.float32, tf.float32, tf.float32, tf.float32,
                                                                tf.int32, tf.int32, tf.float32, tf.float32,
                                                                tf.int32, tf.float32, tf.int32, tf.int32]),
        num_parallel_calls=6)
    train_dataset = train_dataset.prefetch(3)

    test_dataset = tf.data.TextLineDataset(cfg.test_data_file)
    test_dataset = test_dataset.batch(2)
    test_dataset = test_dataset.map(
        lambda x: tf.compat.v1.py_func(get_data, [x], Tout=[tf.float32, tf.float32, tf.float32, tf.float32,
                                                            tf.int32, tf.int32, tf.float32, tf.float32,
                                                            tf.int32, tf.float32, tf.int32, tf.int32]),
        num_parallel_calls=6)
    test_dataset = test_dataset.prefetch(3)

    train_iterator = iter(train_dataset)
    input_data, batch_hm, batch_wh, batch_offset, batch_offset_mask, batch_ind, batch_hm_kp, batch_kps, batch_kps_mask, batch_kp_offset, batch_kp_ind, batch_kp_mask = train_iterator.get_next()
    input_data.set_shape([cfg.batch_size, cfg.input_image_h, cfg.input_image_w, 3])
    batch_hm.set_shape([cfg.batch_size, cfg.output_h, cfg.output_w, cfg.n_classes])
    batch_wh.set_shape([cfg.batch_size, cfg.max_objs, 2])
    batch_offset.set_shape([cfg.batch_size, cfg.max_objs, 2])
    batch_offset_mask.set_shape([cfg.batch_size, cfg.max_objs])
    batch_ind.set_shape([cfg.batch_size, cfg.max_objs])

    batch_hm_kp.set_shape([cfg.batch_size, cfg.output_h, cfg.output_w, cfg.n_kps])
    batch_kps.set_shape([cfg.batch_size, cfg.max_objs, cfg.n_kps, 2])
    batch_kps_mask.set_shape([cfg.batch_size, cfg.max_objs, cfg.n_kps])
    batch_kp_offset.set_shape([cfg.batch_size, cfg.max_objs * cfg.n_kps, 2])
    batch_kp_ind.set_shape([cfg.batch_size, cfg.max_objs * cfg.n_kps])
    batch_kp_mask.set_shape([cfg.batch_size, cfg.max_objs * cfg.n_kps])

    gt = {
        'hm': batch_hm,
        'wh': batch_wh,
        'offset': batch_offset,
        'ind': batch_ind,
        'mask': batch_offset_mask,
        'kp_hm': batch_hm_kp,
        'kps': batch_kps,
        'kp_offset': batch_kp_offset,
        'kp_mask': batch_kp_mask,
        'kp_ind': batch_kp_ind,
        'kps_mask': batch_kps_mask
    }

    test_iterator = iter(test_dataset)
    test_input_data, test_batch_hm, test_batch_wh, test_batch_offset, test_batch_offset_mask, test_batch_ind, test_batch_hm_kp, test_batch_kps, test_batch_kps_mask, test_batch_kp_offset, test_batch_kp_ind, test_batch_kp_mask = test_iterator.get_next()
    test_input_data.set_shape([cfg.batch_size, cfg.input_image_h, cfg.input_image_w, 3])
    test_batch_hm.set_shape([cfg.batch_size, cfg.output_h, cfg.output_w, cfg.n_classes])
    test_batch_wh.set_shape([cfg.batch_size, cfg.max_objs, 2])
    test_batch_offset.set_shape([cfg.batch_size, cfg.max_objs, 2])
    test_batch_offset_mask.set_shape([cfg.batch_size, cfg.max_objs])
    test_batch_ind.set_shape([cfg.batch_size, cfg.max_objs])

    test_batch_hm_kp.set_shape([cfg.batch_size, cfg.output_h, cfg.output_w, cfg.n_kps])
    test_batch_kps.set_shape([cfg.batch_size, cfg.max_objs, cfg.n_kps, 2])
    test_batch_kps_mask.set_shape([cfg.batch_size, cfg.max_objs, cfg.n_kps])
    test_batch_kp_offset.set_shape([cfg.batch_size, cfg.max_objs * cfg.n_kps, 2])
    test_batch_kp_ind.set_shape([cfg.batch_size, cfg.max_objs * cfg.n_kps])
    test_batch_kp_mask.set_shape([cfg.batch_size, cfg.max_objs * cfg.n_kps])

    test_gt = {
        'hm': test_batch_hm,
        'wh': test_batch_wh,
        'offset': test_batch_offset,
        'ind': test_batch_ind,
        'mask': test_batch_offset_mask,
        'kp_hm': test_batch_hm_kp,
        'kps': test_batch_kps,
        'kp_offset': test_batch_kp_offset,
        'kp_mask': test_batch_kp_mask,
        'kp_ind': test_batch_kp_ind,
        'kps_mask': test_batch_kps_mask
    }

    # define model and loss

    model = CenterNet()
    optimizer = tf.keras.optimizers.Adam()

    train_log_losses = {
        "total": tf.keras.metrics.Mean(name='total_loss'),
        "hm":  tf.keras.metrics.Mean(name='hm_loss'),
        "wh":  tf.keras.metrics.Mean(name='wh_loss'),
        "offset":  tf.keras.metrics.Mean(name='offset_loss'),
        "kps":  tf.keras.metrics.Mean(name='kps_loss'),
        "kp_offset":  tf.keras.metrics.Mean(name='kp_offset_loss'),
        "kp_hm":  tf.keras.metrics.Mean(name='kp_hm_loss')
    }

    test_log_losses = {
        "total": tf.keras.metrics.Mean(name='total_loss'),
        "hm": tf.keras.metrics.Mean(name='hm_loss'),
        "wh": tf.keras.metrics.Mean(name='wh_loss'),
        "offset": tf.keras.metrics.Mean(name='offset_loss'),
        "kps": tf.keras.metrics.Mean(name='kps_loss'),
        "kp_offset": tf.keras.metrics.Mean(name='kp_offset_loss'),
        "kp_hm": tf.keras.metrics.Mean(name='kp_hm_loss')
    }

    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), iterator=train_iterator)
    # manager = tf.train.CheckpointManager(ckpt, 'logs/tf_ckpts', max_to_keep=5)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            pred = model(images, training=True)
            losses = compute_loss(pred, labels)

        grads = tape.gradient(losses['total'], model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_log_losses['total'](losses['total'])
        train_log_losses['hm'](losses['hm'])
        train_log_losses['offset'](losses['offset'])
        train_log_losses['wh'](losses['wh'])
        train_log_losses['kp_hm'](losses['kp_hm'])
        train_log_losses['kps'](losses['kps'])
        train_log_losses['kp_offset'](losses['kp_offset'])

    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        pred = model(images, training=False)
        losses = compute_loss(pred, labels)

        test_log_losses['total'](losses['total'])
        test_log_losses['hm'](losses['hm'])
        test_log_losses['offset'](losses['offset'])
        test_log_losses['wh'](losses['wh'])
        test_log_losses['kp_hm'](losses['kp_hm'])
        test_log_losses['kps'](losses['kps'])
        test_log_losses['kp_offset'](losses['kp_offset'])

    train_summary_writer = tf.summary.create_file_writer("logs/{}/train".format(cfg.dataset_name))
    test_summary_writer = tf.summary.create_file_writer("logs/{}/test".format(cfg.dataset_name))
    #
    #
    # ckpt.restore(manager.latest_checkpoint)
    for epoch in range(cfg.epochs):
        # Reset the metrics at the start of the next epoch
        for k in train_log_losses.keys():
            train_log_losses[k].reset_states()
            test_log_losses[k].reset_states()

        for i in range(num_train_batch):
            train_step(input_data, gt)
            with train_summary_writer.as_default():
                for k in train_log_losses.keys():
                    tf.summary.scalar("loss/" + k, train_log_losses[k].result(), step=epoch * num_train_batch + i)

        for i in range(num_test_batch):
            test_step(test_input_data, test_gt)
            with test_summary_writer.as_default():
                for k in test_log_losses.keys():
                    tf.summary.scalar("loss/" + k, test_log_losses[k].result(), step=epoch * num_train_batch + i)

        # if int(ckpt.step) % 1 == 0:
        #     save_path = manager.save()
        #     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1,
                              train_log_losses['total'].result(),
                              test_log_losses['total'].result()))


# # define train op
    #     if cfg.lr_type == "CosineAnnealing":
    #         global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
    #         warmup_steps = tf.constant(cfg.warm_up_epochs * num_train_batch, dtype=tf.float64, name='warmup_steps')
    #         train_steps = tf.constant(cfg.epochs * num_train_batch, dtype=tf.float64, name='train_steps')
    #         learning_rate = tf.cond(
    #             pred=global_step < warmup_steps,
    #             true_fn=lambda: global_step / warmup_steps * cfg.init_lr,
    #             false_fn=lambda: cfg.end_lr + 0.5 * (cfg.init_lr - cfg.end_lr) *
    #                              (1 + tf.cos(
    #                                  (global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
    #         )
    #         global_step_update = tf.compat.v1.assign_add(global_step, 1.0)
    #
    #         optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(losses['total'])
    #         with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
    #             with tf.control_dependencies([optimizer, global_step_update]):
    #                 train_op = tf.no_op()
    #
    #     else:
    #         global_step = tf.Variable(0, trainable=False)
    #         if cfg.lr_type == "exponential":
    #             learning_rate = tf.compat.v1.train.exponential_decay(cfg.lr,
    #                                                                  global_step,
    #                                                                  cfg.lr_decay_steps,
    #                                                                  cfg.lr_decay_rate,
    #                                                                  staircase=True)
    #         elif cfg.lr_type == "piecewise":
    #             learning_rate = tf.compat.v1.train.piecewise_constant(global_step, cfg.lr_boundaries, cfg.lr_piecewise)
    #         optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    #         update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    #         with tf.control_dependencies(update_ops):
    #             train_op = optimizer.minimize(losses['total'], global_step=global_step)
    #
    #
    #
    # saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)
    #
    # with tf.name_scope('summary'):
    #     tf.summary.scalar("learning_rate", learning_rate)
    #     tf.summary.scalar("hm_loss", losses['hm'])
    #     tf.summary.scalar("wh_loss", losses['wh'])
    #     tf.summary.scalar("offset_loss", losses['offset'])
    #     tf.summary.scalar("hm_kp_loss", losses['hm_kp'])
    #     tf.summary.scalar("kp_offset_loss", losses['kp_offset'])
    #     tf.summary.scalar("kps_loss", losses['kps'])
    #     tf.summary.scalar("total_loss", losses['total'])
    #
    #     logdir = "logs/"
    #     if not os.path.exists(logdir):
    #         os.mkdir(logdir)
    #     write_op = tf.compat.v1.summary.merge_all()
    #     summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=sess.graph)
    #
    #     # train
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #
    # for epoch in range(1, 1 + cfg.epochs):
    #     pbar = tqdm(range(num_train_batch))
    #     train_epoch_loss, test_epoch_loss = [], []
    #     sess.run(trainset_init_op)
    #     for i in pbar:
    #         _, summary, train_step_loss, global_step_val = sess.run(
    #             [train_op, write_op, losses['total'], global_step])
    #
    #         train_epoch_loss.append(train_step_loss)
    #         summary_writer.add_summary(summary, global_step_val)
    #         pbar.set_description("train loss: %.2f" % train_step_loss)
    #
    #     sess.run(testset_init_op)
    #     for j in range(num_test_batch):
    #         test_step_loss = sess.run(losses['total'])
    #         test_epoch_loss.append(test_step_loss)
    #
    #     train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
    #     ckpt_file = "./checkpoint/centernet_test_loss=%.4f.ckpt" % test_epoch_loss
    #     log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    #     print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
    #           % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
    #     saver.save(sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    train()
