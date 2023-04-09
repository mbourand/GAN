from keras.layers import Layer
import tensorflow as tf
from keras.initializers import RandomNormal

from settings import *

class ModulatedConv2D(Layer):
	def __init__(self, filters, kernel_size, use_demod=True, **kwargs):
		super().__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.use_demod = use_demod

	def build(self, input_shape):
		image_shape, _ = input_shape
		weight_shape = (self.kernel_size[0], self.kernel_size[1], image_shape[-1], self.filters)

		self.kernels = self.add_weight(
			name='kernels',
			shape=weight_shape,
			initializer=RandomNormal(stddev=1),
			trainable=True
		)

		self.equalizer = tf.sqrt(2 / (weight_shape[0] * weight_shape[1] * weight_shape[2]))

		super().build(input_shape)

	def nhwc_to_nchw(self, inputs):
		return tf.transpose(inputs, [0, 3, 1, 2])

	def nchw_to_nhwc(self, inputs):
		return tf.transpose(inputs, [0, 2, 3, 1])

	def call(self, inputs):
		image, style = inputs

		image = self.nhwc_to_nchw(image)

		weights = self.kernels[None, :, :, :, :] * self.equalizer
		weights *= style[:, None, None, :, None]

		if self.use_demod:
			sigma = tf.math.rsqrt(tf.reduce_sum(tf.square(weights), axis=[1, 2, 3]) + 1e-8)
			weights *= sigma[:, None, None, None, :]

		image = tf.reshape(image, [1, -1, image.shape[2], image.shape[3]])
		w = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])

		conv_result = tf.nn.conv2d(image, w, strides=1, padding='SAME', data_format='NCHW')
		conv_result = tf.reshape(conv_result, [-1, self.filters, image.shape[2], image.shape[3]])
		conv_result = self.nchw_to_nhwc(conv_result)

		return conv_result

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
		})
		return config