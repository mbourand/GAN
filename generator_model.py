from keras.models import Model
from keras.regularizers import *
from keras.layers import *
from settings import *

from EqualizedLRDense import EqualizedLRDense
from ModulatedConv2D import ModulatedConv2D
from EqConv2D import EqConv2D
from BiasLayer import BiasLayer

from settings import *

def style_block(w, x, filters):
	style = EqualizedLRDense(x.shape[-1], bias_init_value=1)(w)
	x = ModulatedConv2D(filters, (KERNEL_SIZE, KERNEL_SIZE), use_demod=True)([x, style])
	# Noise
	x = BiasLayer()(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	return x

def to_rgb(w, x):
	style = EqualizedLRDense(x.shape[-1], bias_init_value=1)(w)
	x = ModulatedConv2D(3, (KERNEL_SIZE, KERNEL_SIZE), use_demod=False)([x, style])
	x = BiasLayer()(x)
	return x

def skip_connection_block(w, x, last_skip_connection, upsample=False):
	skip_connection = to_rgb(w, x)
	if last_skip_connection is not None:
		skip_connection = Add()([skip_connection, last_skip_connection])
	if upsample:
		skip_connection = UpSampling2D(interpolation='bilinear')(skip_connection)
	return skip_connection

def const_input(x):
	x = Lambda(lambda x: tf.ones_like(x))(x)
	x = EqualizedLRDense(IMAGE_MIN_SIZE * IMAGE_MIN_SIZE * FILTERS_MAX, use_bias=False)(x)
	x = Reshape((IMAGE_MIN_SIZE, IMAGE_MIN_SIZE, FILTERS_MAX))(x)
	return x

def get_generator_model():
	input = Input(shape=(1,))
	w_input = Input(shape=(LATENT_DIM,))

	constant = const_input(input)

	x = style_block(w_input, constant, FILTERS_MAX)
	skip_connection = skip_connection_block(w_input, x, None, upsample=True)
	for i in range(BLOCKS_COUNT):
		x = UpSampling2D(interpolation='bilinear')(x)
		filters = max(FILTERS_MIN, FILTERS_MAX // (2**(i + 1)))
		x = style_block(w_input, x, filters)
		x = style_block(w_input, x, filters)
		skip_connection = skip_connection_block(w_input, x, skip_connection, upsample=(i != BLOCKS_COUNT - 1))

	x = Activation('tanh')(skip_connection)

	generator = Model([input, w_input], x)

	return generator
