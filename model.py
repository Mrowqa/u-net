import tensorflow as tf
import data_pipeline as dp
import train_valid_split as tvs
from params import *
import itertools
import logging

log = logging.getLogger('model')


class UNet:
    def __init__(self, layers, training):
        self.sess = None
        self.x = tf.placeholder(tf.float32, [None, None, None, IMAGE_CHANNELS])
        self.y_target = tf.placeholder(tf.uint8, [None, None, None, LABEL_CHANNELS])
        self.y_target_1h = tf.placeholder(tf.float32, [None, None, None, CATEGORIES_CNT])
        self.filename = tf.placeholder(tf.string)

        with tf.device(GPU_ID):
            signal = self.x
            for fno, frame_desc in enumerate(layers):
                with tf.variable_scope('Frame_{}'.format(fno)):
                    for lno, l in enumerate(frame_desc):
                        signal = l.route_signal(signal=signal,
                                                var_name='W{}_{}'.format(fno, lno),
                                                training=training)

            signal = Conv2D(3, CATEGORIES_CNT, False).route_signal(signal=signal,
                                                                   var_name='W_out',
                                                                   training=training)
        self.probs = tf.nn.softmax(signal)

        self.preds = tf.cast(tf.argmax(self.probs, axis=3), tf.uint8)
        self.preds = tf.expand_dims(self.preds, -1)
        # Note batch_size=1 if saving to file!
        self.save_preds_to_file = dp.encode_and_save_to_file(self.filename, self.preds[0])

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=signal, labels=self.y_target_1h))
        self.step_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_target, self.preds), tf.float32))
        self.read_overall_accuracy, self.update_overall_accuracy = tf.metrics.accuracy(self.y_target, self.preds)

        # if training: ??
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        log.debug('List of variables: {}'.format(list(map(lambda x: x.name, tf.global_variables()))))

    def train(self, dataset, mb_size, save_config=None):  # TODO save checkpoints
        train_files = tvs.build_full_paths(dataset, "train")
        # valid_files = tvs.build_full_paths(dataset, "valid")
        with tf.device('/cpu:0'):
            ds = dp.build_train_input_pipeline(train_files, mb_size)
            next_batch = ds.make_one_shot_iterator().get_next()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.sess:
            tf.local_variables_initializer().run()
            if save_config and save_config['initial_load'] is not None:
                saver.restore(self.sess, save_config['initial_load'])
                log.info("Weights restored from {}.".format(save_config['initial_load']))
            else:
                tf.global_variables_initializer().run()
                log.info("Training model from scratch.")
            try:
                for i in itertools.count():
                    img, lbl, lbl_1h = self.sess.run(next_batch)
                    _, loss, acc, acc2 = \
                        self.sess.run([self.train_step, self.loss, self.update_overall_accuracy, self.step_accuracy],
                                      feed_dict={self.x: img, self.y_target: lbl, self.y_target_1h: lbl_1h})
                    log.debug("Loss: {}, Epoch acc: {}, Step acc: {}".format(loss, acc, acc2))
                    if save_config and i % save_config['emergency_after_batches'] == 0:
                        save_path = saver.save(self.sess, save_config['emergency_save'])
                        log.info("Model saved in path: {}".format(save_path))
            except tf.errors.OutOfRangeError:
                pass

            if save_config and save_config['final_save'] is not None:
                save_path = saver.save(self.sess, save_config['final_save'])
                log.info("Model saved in path: {}".format(save_path))
                # note stop dropbox sync, otherwise it blocks renaming file and SILENTLY crashes the saver, same about antivirus
            else:
                log.info("Save directory for the model was not specified.")

    def eval(self):
        pass


class Layer:
    pass


class Conv2D(Layer):
    def __init__(self, patch_edge, channels_out, add_relu=True):
        self.patch_edge = patch_edge
        self.channels_out = channels_out
        self.add_relu = add_relu

    def route_signal(self, signal, var_name, training):
        channels_in = signal.shape[3]
        weights = weight_variable(var_name, [self.patch_edge, self.patch_edge, channels_in, self.channels_out])
        signal = tf.nn.conv2d(signal, weights, strides=[1, 1, 1, 1], padding="SAME")
        signal = tf.layers.batch_normalization(signal, training=training)
        if self.add_relu:
            signal = tf.nn.relu(signal)
        return signal


class Deconv2D_Then_Concat(Layer):
    def __init__(self, patch_edge, channels_out, stack):
        self.patch_edge = patch_edge
        self.channels_out = channels_out
        self.stack = stack

    def route_signal(self, signal, var_name, **_kwargs):
        prev_signal = self.stack.pop()
        channels_in = signal.shape[3]
        weights = weight_variable(var_name, [self.patch_edge, self.patch_edge, self.channels_out, channels_in])
        strides = [1, self.patch_edge, self.patch_edge, 1]
        signal = tf.nn.conv2d_transpose(signal, weights, output_shape=tf.shape(prev_signal), strides=strides,
                                        padding="SAME")
        signal = tf.concat([signal, prev_signal], axis=3)
        return signal


class Push_Then_MaxPool(Layer):
    def __init__(self, patch_edge, stack):
        self.patch_edge = patch_edge
        self.stack = stack

    def route_signal(self, signal, **_kwargs):
        self.stack.append(signal)
        kernel_strides = [1, self.patch_edge, self.patch_edge, 1]
        signal = tf.nn.max_pool(signal, ksize=kernel_strides, strides=kernel_strides, padding='SAME')
        return signal


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())  # Xavier Glorot


# ------------------------- exec stuff ----------------
def create_model(name, training=False):
    stack = []

    def double_conv(channels):
        return [
            Conv2D(3, channels),
            Conv2D(3, channels),
        ]

    def block_down(channels):
        return double_conv(channels) + [
            Push_Then_MaxPool(2, stack),
        ]

    def block_up(channels):
        return [
            Deconv2D_Then_Concat(2, channels, stack),
        ] + double_conv(channels)

    layers = [
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

    with tf.variable_scope("UNet_{}".format(name)):
        return UNet(layers, training)
