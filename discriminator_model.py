from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import *
from settings import *
from keras.regularizers import *

def get_discriminator_model():
	discriminator = Sequential([
		Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL)),
		Conv2D(3, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(LEAKY_RELU_ALPHA),
		BatchNormalization(),
		Conv2D(128, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(LEAKY_RELU_ALPHA),
		BatchNormalization(),
		Conv2D(256, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(LEAKY_RELU_ALPHA),
		BatchNormalization(),
		Conv2D(512, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(LEAKY_RELU_ALPHA),

		Flatten(),
		Dense(1, kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV), activation='sigmoid'),
	])

	return discriminator
