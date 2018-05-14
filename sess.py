# as a prefix, for python -i sess.py
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

