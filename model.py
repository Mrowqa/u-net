import tensorflow as tf
import data_pipeline as dp
import train_valid_split as tvs
from params import *
from utils import *
import itertools
import logging
import os
import numpy as np

log = logging.getLogger('model')


class UNet:
    def __init__(self, layers, name, training):
        self.name = name
        self.training = training
        self.sess = None
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, None, None, IMAGE_CHANNELS])
            self.y_target = tf.placeholder(tf.uint8, [None, None, None, LABEL_CHANNELS])
            self.filename = tf.placeholder(tf.string)

        with tf.device(GPU_ID):
            signal = self.x
            for fno, frame_desc in enumerate(layers):
                with tf.variable_scope('Frame_{}'.format(fno)):
                    for lno, l in enumerate(frame_desc):
                        signal = l.route_signal(signal=signal,
                                                var_name='W{}_{}'.format(fno, lno),
                                                training=training)

            with tf.name_scope('preparing_readout'):
                signal = Conv2D(3, CATEGORIES_CNT, False).route_signal(signal=signal,
                                                                       var_name='W_out',
                                                                       training=training)

        with tf.name_scope('prediction'):
            self.probs = tf.nn.softmax(signal)
            self.preds = tf.cast(tf.argmax(self.probs, axis=3), tf.uint8)
            self.preds = tf.expand_dims(self.preds, -1)

        with tf.name_scope('saving_output_to_file'):
            # Note batch_size=1 if saving to file!
            self.save_preds_to_file = dp.encode_and_save_to_file(self.filename, self.preds[0])

        with tf.name_scope('loss'):
            label_1hot = tf.one_hot(tf.squeeze(self.y_target, axis=3), CATEGORIES_CNT, axis=-1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=signal, labels=label_1hot))
        with tf.name_scope('accuracy'):
            self.step_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_target, self.preds), tf.float32))
            self.read_overall_accuracy, self.update_overall_accuracy = tf.metrics.mean(self.step_accuracy)

        with tf.name_scope('summaries'):
            sloss = tf.summary.scalar('loss', self.loss)
            sacc = tf.summary.scalar('accuracy', self.step_accuracy)
            sacc2 = tf.summary.scalar('moving_mean_accuracy', self.read_overall_accuracy)
            self.loss_acc_summaries = tf.summary.merge([sloss, sacc, sacc2])

            zero = np.array([[[[0, 0, 0]]]], dtype=np.uint8)
            self.y_target_color = tf.placeholder_with_default(zero, [None, None, None, IMAGE_CHANNELS])
            self.preds_color = tf.placeholder_with_default(zero, [None, None, None, IMAGE_CHANNELS])
            simg = tf.summary.image('image', self.x)
            strg = tf.summary.image('target', self.y_target_color)
            sprd = tf.summary.image('prediction', self.preds_color)
            self.images_summaries = tf.summary.merge([simg, strg, sprd])

        with tf.name_scope('training_step'):
            if training:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        log.debug('List of variables: {}'.format(list(map(lambda x: x.name, tf.global_variables()))))

    def train(self, dataset, mb_size, save_config=None):
        assert self.training
        log.info('Training.')

        train_files = tvs.build_full_paths(dataset, "train")[:10]  # TODO hack
        with tf.device('/cpu:0'):
            ds = dp.build_train_input_pipeline(train_files, mb_size)
            next_batch = ds.make_one_shot_iterator().get_next()
        saver = tf.train.Saver()

        summary_dir = os.path.join(SUMMARIES_LOG_DIR, 'train/', self.name + '-' + file_formatted_now())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.sess, \
                tf.summary.FileWriter(summary_dir, self.sess.graph) as s_writer:
            tf.local_variables_initializer().run()
            if save_config and save_config['initial_load'] is not None:
                saver.restore(self.sess, save_config['initial_load'])
                log.info("Weights restored from {}.".format(save_config['initial_load']))
            else:
                tf.global_variables_initializer().run()
                log.info("Training model from scratch.")
            try:
                for i in itertools.count():
                    img, lbl = self.sess.run(next_batch)
                    _, loss, acc, acc2, summaries = \
                        self.sess.run([self.train_step, self.loss, self.update_overall_accuracy, self.step_accuracy,
                                       self.loss_acc_summaries],
                                      feed_dict={self.x: img, self.y_target: lbl})
                    log.debug("Loss: {}, Epoch acc: {}, Step acc: {}".format(loss, acc, acc2))
                    s_writer.add_summary(summaries, i)
                    if save_config and i % save_config['emergency_after_batches'] == 0:
                        save_path = saver.save(self.sess, save_config['emergency_save'])
                        log.info("At step {}: Model saved in path: {}".format(i, save_path))
            except tf.errors.OutOfRangeError:
                pass

            if save_config and save_config['final_save'] is not None:
                save_path = saver.save(self.sess, save_config['final_save'])
                log.info("Model saved in path: {}".format(save_path))
                # note stop dropbox sync, otherwise it blocks renaming file and SILENTLY crashes the saver, same about antivirus
            else:
                log.info("Save directory for the model was not specified.")

    def validate(self, dataset, saved_model_weights):
        # assert not self.training
        log.info('Validating.')

        valid_files = tvs.build_full_paths(dataset, "valid")[:10]  # TODO hack
        with tf.device('/cpu:0'):
            ds = dp.build_evaluate_input_pipeline(valid_files, for_validation=True)
            next_batch = ds.make_one_shot_iterator().get_next()
        saver = tf.train.Saver()

        summary_dir = os.path.join(SUMMARIES_LOG_DIR, 'validate/', self.name + '-' + file_formatted_now())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.sess, \
                tf.summary.FileWriter(summary_dir, self.sess.graph) as s_writer:
            tf.local_variables_initializer().run()
            saver.restore(self.sess, saved_model_weights)
            log.info("Weights restored from {}.".format(saved_model_weights))
            try:
                for i in itertools.count(start=1):
                    case_list = self.sess.run(next_batch)
                    for j, (img, lbl) in enumerate(case_list):
                        step = i + (j / len(case_list))
                        img_chks, lbl_chks = dp.split_into_chunks(img), dp.split_into_chunks(lbl)
                        log.debug("Step {} | Shape: {} split into {} chunks.".format(step, img.shape, len(img_chks)))
                        losses = []
                        all_preds = []
                        for chimg, chlbl in zip(img_chks, lbl_chks):
                            loss, pred = \
                                self.sess.run([self.loss, self.preds],
                                              feed_dict={self.x: chimg, self.y_target: chlbl})
                            losses.append(loss)
                            all_preds.append(pred)
                        overall_pred = dp.merge_chunks(all_preds, lbl.shape)
                        loss = np.mean(losses)
                        acc, acc2, summaries, img_summaries = \
                            self.sess.run([self.update_overall_accuracy, self.step_accuracy,
                                           self.loss_acc_summaries, self.images_summaries],
                                          feed_dict={self.x: img,
                                                     self.preds: overall_pred,
                                                     self.loss: loss,
                                                     self.y_target: lbl})
                        log.debug("Loss: {}, Mov mean acc: {}, Step acc: {}".format(loss, acc, acc2))
                        s_writer.add_summary(summaries, step)
                        if i % 100 == 0:
                            # note: coloring is very slow
                            log.debug("Step {}: adding visualizations".format(step))
                            img_summaries = self.sess.run(self.images_summaries,
                                                          feed_dict={self.x: img,
                                                                     self.preds_color: dp.color_labels(overall_pred),
                                                                     self.y_target_color: dp.color_labels(lbl)})
                            s_writer.add_summary(img_summaries, i+j)  # hack, assuming this won't cause anything to overlap
            except tf.errors.OutOfRangeError:
                pass

    def evaluate(self, dataset, saved_model_weights):
        assert not self.training
        log.info('Evaluating.')

        eval_files = dataset[:10]  # TODO hack
        with tf.device('/cpu:0'):
            ds = dp.build_evaluate_input_pipeline(eval_files)
            next_batch = ds.make_one_shot_iterator().get_next()
        saver = tf.train.Saver()

        output_dir = os.path.join(EVAL_OUTPUT_DIR, self.name + '-' + file_formatted_now() + '/')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as self.sess:
            tf.local_variables_initializer().run()
            saver.restore(self.sess, saved_model_weights)
            log.info("Weights restored from {}.".format(saved_model_weights))
            try:
                for i in itertools.count(start=1):
                    img, basename = self.sess.run(next_batch)
                    basename = basename[0].decode()  # unpack the batch
                    img_chks = dp.split_into_chunks(img)
                    log.debug("Step {} | Shape: {} split into {} chunks.".format(i, img.shape, len(img_chks)))
                    all_preds = []
                    for chimg in img_chks:
                        pred = self.sess.run(self.preds, feed_dict={self.x: chimg})
                        all_preds.append(pred)
                    lbl_shape = [*img.shape[:3], LABEL_CHANNELS]
                    overall_pred = dp.merge_chunks(all_preds, lbl_shape)
                    output_filename = os.path.join(output_dir, basename + '-out.png')
                    log.debug("Saving to file {}.".format(output_filename))
                    self.sess.run(self.save_preds_to_file, feed_dict={self.preds: overall_pred,
                                                                      self.filename: output_filename})
            except tf.errors.OutOfRangeError:
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
        signal = tf.layers.batch_normalization(signal, momentum=0.9, training=training)
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
def create_model(name, sf=1, reuse_vars=False, training=False):
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
        block_down(16*sf),
        block_down(32*sf),
        block_down(64*sf),
        block_down(128*sf),
        double_conv(256*sf),
        block_up(128*sf),
        block_up(64*sf),
        block_up(32*sf),
        block_up(16*sf),
    ]

    with tf.variable_scope("UNet_{}".format(name), reuse=reuse_vars):
        return UNet(layers, name, training)
