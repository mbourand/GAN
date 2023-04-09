from keras.layers import Layer
import tensorflow as tf
from keras.activations import *
from keras.initializers import Constant, RandomNormal
from settings import *

class EqDense(Layer):
	def __init__(self, units, bias_init_value=0, use_bias=True, learning_rate_scale=1, **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.bias_init_value = bias_init_value
		self.use_bias = use_bias
		self.learning_rate_scale = learning_rate_scale

	def build(self, input_shape):
		self.equalized_weights = self.add_weight(
			name='equalized_weights',
			shape=(input_shape[1], self.units),
			initializer=RandomNormal(stddev=1 / self.learning_rate_scale),
			trainable=True
		)

		if self.use_bias:
			self.equalized_biases = self.add_weight(
				name='equalized_biases',
				shape=(self.units,),
				initializer=Constant(self.bias_init_value / self.learning_rate_scale),
				trainable=True
			)

		self.equalizer = tf.sqrt(2 / input_shape[1])

		super().build(input_shape)

	def call(self, inputs):
		res = tf.matmul(inputs, self.equalized_weights * self.equalizer * self.learning_rate_scale)
		if self.use_bias:
			return res + (self.equalized_biases[None, :] * self.learning_rate_scale)
		return res

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'units': self.units,
			'bias_init_value': self.bias_init_value,
			'use_bias': self.use_bias,
			'learning_rate_scale': self.learning_rate_scale
		})
		return config