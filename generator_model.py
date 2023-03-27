from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.regularizers import *
from keras.layers import *
from settings import *

from PixelNormalization import PixelNormalization
from EqualizedLRDense import EqualizedLRDense

def get_generator_model():
	generator = Sequential([
		Input(shape=(LATENT_DIM,)),
		EqualizedLRDense(4 * 4 * 1024),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		Reshape((4, 4, 1024)),
		PixelNormalization(),
		BatchNormalization(),
		
		Conv2DTranspose(512, KERNEL_SIZE, strides=2, padding='same'),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		PixelNormalization(),
		BatchNormalization(),

		Conv2DTranspose(256, KERNEL_SIZE, strides=2, padding='same'),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		PixelNormalization(),
		BatchNormalization(),

		Conv2DTranspose(128, KERNEL_SIZE, strides=2, padding='same'),
		LeakyReLU(alpha=LEAKY_RELU_ALPHA),
		PixelNormalization(),
		BatchNormalization(),

		Conv2DTranspose(3, KERNEL_SIZE, strides=2, padding='same'),
		Activation('tanh')
	])
	return generator
