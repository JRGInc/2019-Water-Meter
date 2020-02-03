# some training parameters
EPOCHS = 200
BATCH_SIZE = 6
NUM_CLASSES = 10
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
CHANNELS = 1
save_model_dir = "saved_model/"
save_every_n_epoch = 10
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
cache_dir = "cache/"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2

# choose a network
# 1: InceptionV4, 2: InceptionResNetV1, 3: InceptionResNetV2
model_index = 1

