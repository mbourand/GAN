from keras.layers import Layer
import tensorflow as tf
from keras.initializers import Constant, RandomNormal

from settings import *

class EqConv2D(Layer):
	def __init__(self, filters, kernel_size, strides=1, bias_init_value=0, use_bias=True, **kwargs):
		super().__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.bias_init_value = bias_init_value
		self.use_bias = use_bias

	def build(self, input_shape):
		weight_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters)

		self.equalized_weights = self.add_weight(
			name='equalized_weights',
			shape=weight_shape,
			initializer=RandomNormal(stddev=1),
			trainable=True
		)
		if self.use_bias:
			self.equalized_biases = self.add_weight(
				name='equalized_biases',
				shape=(self.filters,),
				initializer=Constant(self.bias_init_value),
				trainable=True
			)

		self.equalizer = tf.sqrt(2 / (weight_shape[0] * weight_shape[1] * weight_shape[2]))

		super().build(input_shape)

	def call(self, inputs):
		conv_result = tf.nn.conv2d(
			inputs,
			self.equalized_weights * self.equalizer,
			strides=(1, self.strides, self.strides, 1),
			padding='SAME'
		)

		if not self.use_bias:
			return conv_result
		return conv_result + self.equalized_biases[None, None, None, :]

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			'bias_init_value': self.bias_init_value,
			'use_bias': self.use_bias
		})
		return config