from keras.models import Model
from keras.layers import *
from settings import *
from keras.regularizers import *
from keras.activations import *

from EqConv2D import EqConv2D
from EqDense import EqDense

from MinibatchStandardDeviation import MinibatchStandardDeviation

import tensorflow as tf

def from_rgb(x, filters):
	x = EqConv2D(filters, (1, 1))(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	return x

def residual_block(x, filters):
	residual = EqConv2D(filters, (1, 1), use_bias=False)(x)
	residual = AveragePooling2D(padding='same')(residual)
	return residual

def conv_block(x, filters):
	x = EqConv2D(filters, (KERNEL_SIZE, KERNEL_SIZE))(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	x = EqConv2D(filters, (KERNEL_SIZE, KERNEL_SIZE))(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	x = AveragePooling2D(padding='same')(x)
	return x

def get_filters(stage):
	return min(FILTERS_MAX, FILTERS_MIN * (2**stage))

def get_discriminator_model():
	input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
	x = from_rgb(input, get_filters(0))

	for i in range(BLOCKS_COUNT - 1):
		filters = get_filters(i)

		residual = residual_block(x, filters)
		x = conv_block(x, filters)

		x = Add()([x, residual])
		x = Lambda(lambda x: x / math.sqrt(2))(x)

	x = MinibatchStandardDeviation()(x)
	x = EqConv2D(get_filters(BLOCKS_COUNT), (KERNEL_SIZE, KERNEL_SIZE))(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	x = Flatten()(x)
	x = EqDense(get_filters(BLOCKS_COUNT))(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	x = EqDense(1)(x)

	model = Model(input, x)
	return model