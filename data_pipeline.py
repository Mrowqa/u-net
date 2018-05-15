import tensorflow as tf
from params import *


# TODO some tf name scopes?

# https://cs230-stanford.github.io/tensorflow-input-data.html
# https://www.tensorflow.org/versions/master/performance/datasets_performance
# TODO with device CPU
def build_input_pipeline(images_files, labels_files, is_training):
    # TODO I/O interleaving?
    # TODO cycle!
    dataset = tf.data.Dataset.from_tensor_slices((images_files, labels_files))
    dataset = dataset.shuffle(len(images_files))
    dataset = dataset.map(parse_function, num_parallel_calls=PARALLEL_CALLS)
    #if is_training:
    #    dataset = dataset.map(train_preprocess, num_parallel_calls=PARALLEL_CALLS)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(HOW_MANY_PREFETCH)
    return dataset
    # How to use:
    # images, labels = iterator.get_next()
    # iterator_init_op = iterator.initializer
    #
    # inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}


def parse_function(image_file, label_file):
    image_string = tf.read_file(image_file)
    label_string = tf.read_file(label_file)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3, ratio=1)
    orig_label = label = tf.image.decode_png(label_string, channels=LABEL_CHANNELS)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.shape(image)  # [height * width * channels]

    def width_gt(image, label, shape):
        size = [tf.cast(IMG_EDGE_SIZE * shape[0] / shape[1], tf.int32),  # height
                tf.cast(IMG_EDGE_SIZE, tf.int32)]  # width
        mask = tf.ones(size + [1], dtype=tf.bool)
        image = tf.image.resize_images(image, size)
        label = tf.image.resize_images(label, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        left_h = IMG_EDGE_SIZE - size[0]
        left_h_up = tf.cast(left_h / 2, tf.int32)
        left_h_down = left_h - left_h_up
        padding = [[left_h_up, left_h_down], [0, 0], [0, 0]]
        return image, label, mask, padding

    image, label, mask, padding = tf.cond(shape[1] >= shape[0],
                                          lambda: width_gt(image, label, shape),
                                          lambda: width_gt(image, label, shape))  # todo analogus for heigth

    image = tf.pad(image, padding, constant_values=0)
    label = tf.pad(label, padding, constant_values=CATEGORIES_CNT)  # first out of range
    mask = tf.pad(mask, padding, constant_values=False)

    label_1hot = tf.one_hot(label, CATEGORIES_CNT, axis=-1)

    return image, label_1hot, mask, orig_label, shape, padding


#def remap_values(tensor, mapping):
#    for k, v in mapping.items():
#        cond = tf.equal(tensor, k)
#        tensor = tf.where(cond, tf.ones_like(tensor) * v, tensor)
#    return tensor


# def train_preprocess(image, label):
    # TODO
    # image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    # return image, label


def save_to_file(filename, label, padding, orig_size):
    shape = tf.shape(label)
    label = label[padding[0][0]:shape[0]-padding[0][1], padding[1][0]:shape[1]-padding[1][1]]
    # label = tf.reshape(label, shape + [1])  # let's assume we have this one extra channel
    label = tf.image.resize_images(label, orig_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    data_str = tf.image.encode_png(label)
    return tf.write_file(filename, data_str)


#################################
def test():
    im, l1h, _, _, sh, pad = parse_function(R"D:\students\dnn\assignment2\training\images\00qclUcInksIYnm19b1Xfw.jpg",
                                            "00qclUcInksIYnm19b1Xfw.png")  #R"D:\students\dnn\assignment2\training\labels\00qclUcInksIYnm19b1Xfw.png")
    sav = save_to_file("xd.png", tf.cast(tf.argmax(l1h, axis=3), dtype=tf.uint8), pad, sh[:2])
    return sav


def work(args):
    lbs, verbose = args
    from sess import sess
    vals = set()
    for i, l in enumerate(lbs):
        if verbose:
            print('{} / {}, {} '.format(i, 18000, len(vals)))
        l = tf.image.decode_png(tf.read_file(l))
        l = sess.run(l)
        assert (l.shape[2] == 1)
        vals = vals | set(l.flatten())
        if len(vals) == CATEGORIES_CNT:
            break
    return vals
def helper():
    import glob
    from multiprocessing import Pool, TimeoutError
    lbs = glob.glob(R"D:\students\dnn\assignment2\training\labels\*.png")
    lbs.sort()


    pool = Pool(processes=6)
    jobs = [(lbs[:100], True), (lbs[100:200], False),
            (lbs[200:300], False), (lbs[300:400], False),
            (lbs[400:500], False), (lbs[500:600], False),
            ]
    vals = pool.map(work, jobs)
    vals = vals[0] | vals[1] | vals[2] | vals[3] | vals[4] | vals[5]
    print('cnt:', len(vals))
    print('vals:', vals)
