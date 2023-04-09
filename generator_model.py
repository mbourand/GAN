from keras.models import Model
from keras.regularizers import *
from keras.layers import *
from settings import *

from EqDense import EqDense
from ModulatedConv2D import ModulatedConv2D
from NoiseLayer import NoiseLayer
from BiasLayer import BiasLayer

from settings import *

def style_block(w, x, noise, filters):
	style = EqDense(x.shape[-1], bias_init_value=1)(w)
	x = ModulatedConv2D(filters, (KERNEL_SIZE, KERNEL_SIZE), use_demod=True)([x, style])
	cropped_noise = Cropping2D((noise.shape[1] - x.shape[1]) // 2)(noise)
	x = NoiseLayer()([x, cropped_noise])
	x = BiasLayer()(x)
	x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
	return x

def to_rgb(w, x):
	style = EqDense(x.shape[-1], bias_init_value=1)(w)
	x = ModulatedConv2D(3, (KERNEL_SIZE, KERNEL_SIZE), use_demod=False)([x, style])
	x = BiasLayer()(x)
	return x

def const_input(x):
	x = Lambda(lambda x: tf.ones_like(x))(x)
	x = EqDense(IMAGE_MIN_SIZE * IMAGE_MIN_SIZE * FILTERS_MAX, use_bias=False)(x)
	x = Reshape((IMAGE_MIN_SIZE, IMAGE_MIN_SIZE, FILTERS_MAX))(x)
	return x

def get_filters(stage):
	return max(FILTERS_MIN, FILTERS_MAX // (2**stage))

def get_generator_model():
	input = Input(shape=(1,))
	w_inputs = [Input(shape=(LATENT_DIM,)) for _ in range(BLOCKS_COUNT)]
	noise_inputs = [Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range(BLOCKS_COUNT * 2 + 1)]

	constant = const_input(input)

	x = style_block(w_inputs[0], constant, noise_inputs[0], FILTERS_MAX)
	last_skip_connection = to_rgb(w_inputs[0], x)

	for i in range(BLOCKS_COUNT):
		filters = get_filters(i)

		x = UpSampling2D(interpolation='bilinear')(x)
		x = style_block(w_inputs[i], x, noise_inputs[i * 2 + 1], filters)
		x = style_block(w_inputs[i], x, noise_inputs[i * 2 + 2], filters)

		skip_connection = to_rgb(w_inputs[i], x)
		last_skip_connection = UpSampling2D(interpolation='bilinear')(last_skip_connection)
		last_skip_connection = Add()([last_skip_connection, skip_connection])

	x = Activation('tanh')(last_skip_connection)
	return Model([input] + w_inputs + noise_inputs, x)
