# General
BUFFER_SIZE = 500
SEED = 42

# Dataset
DATASET_PATH = 'dataset/'

# Model train settings
IMAGE_SIZE = 64
IMAGE_CHANNEL = 3
BATCH_SIZE = 32
EPOCHS = 300

# Model hyperparameters
LATENT_DIM = 128
RANDOM_NORMAL_STD_DEV = 0.02
LEAKY_RELU_ALPHA = 0.2
KERNEL_SIZE = 5

ADAM_LEARNING_RATE = 0.0002
ADAM_BETA1 = 0.5

L2_REGULARIZATION = 0.02

SMOOTH = 0.2

# Image Saving Callback
IMAGE_OUTPUT_PATH = 'bce_images_26032023/'
SAVE_IMAGE_FREQUENCY = 500
MARGIN = IMAGE_SIZE // 8
OUTPUT_SHAPE = (4, 4)

MODEL_OUTPUT_PATH = 'bce_model_26032023/'
