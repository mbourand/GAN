from keras.models import Model
from keras.layers import *
from settings import *
from keras.regularizers import *
from keras.activations import *

from EqConv2D import EqConv2D

from MinibatchStandardDeviation import MinibatchStandardDeviation

def from_rgb(x, filters):
	x = EqConv2D(filters, (KERNEL_SIZE, KERNEL_SIZE), use_demod=False)(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	return x

def get_discriminator_model():
	input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
	x = from_rgb(input, FILTERS_MIN)
	residual = x
	for i in range(BLOCKS_COUNT):
		filters = min(FILTERS_MAX, FILTERS_MIN * (2**i))
		x = EqConv2D(filters, (KERNEL_SIZE, KERNEL_SIZE))(x)
		x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
		x = EqConv2D(filters, (KERNEL_SIZE, KERNEL_SIZE))(x)
		x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
		x = AveragePooling2D()(x)
		residual = AveragePooling2D()(residual)
		x = Add()([x, residual])
		residual = x

	model = Model(input, x)
	return model