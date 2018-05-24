GPU_ID = '/gpu:0'
SUMMARIES_LOG_DIR = 'summaries/'  # TODO as param?
EVAL_OUTPUT_DIR = 'eval_output/'
HOW_MANY_PREFETCH = 2
PARALLEL_CALLS = 4
TRAIN_IMG_EDGE_SIZE = 1536  # 2048  # test it
#  512  -- 0.9662759286385996
# 1024  -- 0.9845252143012153
# 2048  -- 0.9945694534866898
EVAL_IMG_SIZE = (1536, 2048)  # (2448, 3264) # (1250, 1650) #(1080, 1920)  # height x width  # better way: something 3:4
EVAL_MARGIN = 140  # 1 +4)*2 +4)*2 ... +4)*2
MAX_ROT_IN_RAD = 0.5  # around [-30, 30] degrees
OVERSIZE_FACTOR = 1.2  # related to rotation and cropping
IMAGE_CHANNELS = 3
LABEL_CHANNELS = 1
CATEGORIES_CNT = 66
