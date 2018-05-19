import tensorflow as tf
from params import *


# https://cs230-stanford.github.io/tensorflow-input-data.html
# https://www.tensorflow.org/versions/master/performance/datasets_performance
def build_train_input_pipeline(images_labels_files, mb_size):
    # TODO I/O interleaving?
    dataset = tf.data.Dataset.from_tensor_slices(images_labels_files)
    dataset = dataset.shuffle(len(images_labels_files))
    dataset = dataset.map(load_train_data, num_parallel_calls=PARALLEL_CALLS)
    dataset = dataset.map(train_data_augmentation, num_parallel_calls=PARALLEL_CALLS)
    dataset = dataset.batch(mb_size)
    dataset = dataset.prefetch(HOW_MANY_PREFETCH)
    return dataset


def build_evaluate_input_pipeline(images_labels_files):
    # TODO I/O interleaving?
    dataset = tf.data.Dataset.from_tensor_slices(images_labels_files)
    dataset = dataset.shuffle(len(images_labels_files))
    dataset = dataset.map(load_evaluate_data, num_parallel_calls=PARALLEL_CALLS)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(HOW_MANY_PREFETCH)
    return dataset


def load_train_data(image_label_file):
    image_string = tf.read_file(image_label_file[0])
    label_string = tf.read_file(image_label_file[1])

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=IMAGE_CHANNELS, ratio=1)
    label = tf.image.decode_png(label_string, channels=LABEL_CHANNELS)

    def scale_up(img, lbl, min_edge_val):
        shape = tf.cast(tf.shape(img), tf.float64)
        ratio = TRAIN_IMG_EDGE_SIZE / min_edge_val
        new_height = tf.maximum(TRAIN_IMG_EDGE_SIZE, tf.cast(shape[0] * ratio, tf.int32))
        new_width = tf.maximum(TRAIN_IMG_EDGE_SIZE, tf.cast(shape[1] * ratio, tf.int32))
        new_size = [new_height, new_width]
        img = tf.image.resize_images(img, new_size, method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        lbl = tf.image.resize_images(lbl, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        return tf.cast(img, tf.uint8), lbl

    min_edge = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    image, label = tf.cond(min_edge < TRAIN_IMG_EDGE_SIZE,
                           lambda: scale_up(image, label, min_edge),
                           lambda: (image, label))

    image_label = tf.concat([image, label], axis=2)
    image_label = tf.random_crop(image_label,
                                 [TRAIN_IMG_EDGE_SIZE, TRAIN_IMG_EDGE_SIZE, IMAGE_CHANNELS + LABEL_CHANNELS])
    image, label = tf.split(image_label, [IMAGE_CHANNELS, LABEL_CHANNELS], axis=2)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    label_1hot = tf.one_hot(tf.squeeze(label, axis=2), CATEGORIES_CNT, axis=-1)  # TODO move it to UNet.__init__()

    return image, label, label_1hot


def train_data_augmentation(*args):
    return args
    # TODO
    # image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    # return image, label


def flip_left_right(image):
    # image is [height x width x channels]
    return tf.reverse(image, axis=[1])  # reverse columns


def load_evaluate_data(image_label_file):
    image_string = tf.read_file(image_label_file[0])
    image = tf.image.decode_jpeg(image_string, channels=IMAGE_CHANNELS, ratio=1)
    image = tf.image.convert_image_dtype(image, tf.float32)

    if image_label_file[1] is not None:
        label_string = tf.read_file(image_label_file[1])
        label = tf.image.decode_png(label_string, channels=LABEL_CHANNELS)
        label_1hot = tf.one_hot(tf.squeeze(label, axis=2), CATEGORIES_CNT, axis=-1)  # TODO move it to UNet.__init__()
    else:
        label = label_1hot = None

    return image, label, label_1hot


def encode_and_save_to_file(filename, preds):
    data_str = tf.image.encode_png(preds)
    return tf.write_file(filename, data_str)
