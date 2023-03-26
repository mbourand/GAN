from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.regularizers import *
from keras.layers import *
from settings import *
from PixelNormalization import PixelNormalization

def get_generator_model():
	generator = Sequential([
		Dense(4 * 4 * 1024, input_shape=(LATENT_DIM,), kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		Reshape((4, 4, 1024)),
		PixelNormalization(),
		
		Conv2DTranspose(512, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		PixelNormalization(),

		Conv2DTranspose(256, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		PixelNormalization(),

		Conv2DTranspose(128, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV)),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		PixelNormalization(),

		Conv2DTranspose(3, KERNEL_SIZE, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV), activation='tanh')
	])
	return generator
