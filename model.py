import tensorflow as tf
import data_pipeline as dp
import train_valid_split as tvs
from params import *
import os


class UNet:
    def __init__(self, layers):
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
                        signal = l.route_signal(signal, 'W{}_{}'.format(fno, lno))

            signal = Conv2D(3, CATEGORIES_CNT, False).route_signal(signal, 'W_out')
            self.probs = tf.nn.softmax(signal - self.y_target_1h)

            self.preds = tf.cast(tf.argmax(self.probs, axis=3), tf.uint8)
            self.preds = tf.expand_dims(self.preds, -1)
            # Note batch_size=1 if saving to file!
            self.save_preds_to_file = dp.encode_and_save_to_file(self.filename, self.preds[0])

            #self.loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.y_target_1h, self.probs))
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=signal, labels=self.y_target_1h))
            # for debugging
            self.train_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_target, self.preds), tf.float32))
            self.accuracy, self.overall_accuracy = tf.metrics.accuracy(self.y_target, self.preds)

            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train(self, dataset, n_batches, mb_size, saved_model_path=None):  # TODO save checkpoints
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
            if saved_model_path is not None and os.path.exists(saved_model_path + '.index'):
                saver.restore(self.sess, saved_model_path)
                print("Weights restored from {}.".format(saved_model_path))
            else:
                tf.global_variables_initializer().run()
                print("Training model from scratch.")
            try:
                dataset_size = len(dataset["train"])  # not dividing by mb_size, it will be caught by stop iteration
                for i in range(n_batches or dataset_size):
                    img, lbl, lbl_1h = self.sess.run(next_batch)
                    _, loss, acc, acc2 = \
                        self.sess.run([self.train_step, self.loss, self.overall_accuracy, self.train_accuracy],
                                      feed_dict={self.x: img, self.y_target: lbl, self.y_target_1h: lbl_1h})
                    print("Loss: {}, Accuracy: {}, Train Acc: {}".format(loss, acc, acc2))
                    if i % 100 == 0:
                        print("Validation not implemented!")  # TODO eval?
            except tf.errors.OutOfRangeError:
                pass

            if saved_model_path:
                save_path = saver.save(self.sess, saved_model_path)
                print("Model saved in path: {}".format(save_path))
                # note stop dropbox sync, otherwise it blocks renaming file and SILENTLY crashes the saver, same about antivirus
            else:
                print("Save directory for the model was not specified.")

    def eval(self):
        pass


class Layer:
    pass


class Conv2D(Layer):
    def __init__(self, patch_edge, channels_out, add_relu=True):
        self.patch_edge = patch_edge
        self.channels_out = channels_out
        self.add_relu = add_relu

    def route_signal(self, signal, var_name):
        channels_in = signal.shape[3]
        weights = weight_variable(var_name, [self.patch_edge, self.patch_edge, channels_in, self.channels_out])
        signal = tf.nn.conv2d(signal, weights, strides=[1, 1, 1, 1], padding="SAME")
        if self.add_relu:
            signal = tf.nn.relu(signal)
        return signal


class Deconv2D_Then_Concat(Layer):
    def __init__(self, patch_edge, channels_out, stack):
        self.patch_edge = patch_edge
        self.channels_out = channels_out
        self.stack = stack

    def route_signal(self, signal, var_name):
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

    def route_signal(self, signal, _):
        self.stack.append(signal)
        kernel_strides = [1, self.patch_edge, self.patch_edge, 1]
        signal = tf.nn.max_pool(signal, ksize=kernel_strides, strides=kernel_strides, padding='SAME')
        return signal


def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())  # Xavier Glorot


# ------------------------- exec stuff ----------------
def create_model(name):
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

    layers = [  # TODO note: check what are intermediate sizes
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
        return UNet(layers)


def main():
    mb_size = 8
    n_batch = 200

    unet = create_model("test")
    dataset = tvs.select_part_for_training(tvs.load_from_file("file.json"), 0)
    unet.train(dataset, n_batch, mb_size, "models/test.ckpt")
