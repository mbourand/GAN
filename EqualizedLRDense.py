from keras.layers import Layer
import tensorflow as tf
from keras.activations import *
from keras.initializers import Constant, RandomNormal
from settings import *

class EqualizedLRDense(Layer):
	def __init__(self, units, bias_init_value=0, **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.bias_init_value = bias_init_value

	def build(self, input_shape):
		self.equalized_weights = self.add_weight(
			name='equalized_weights',
			shape=(input_shape[1], self.units),
			initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV),
			trainable=True
		)

		self.equalized_biases = self.add_weight(
			name='equalized_biases',
			shape=(self.units,),
			initializer=Constant(self.bias_init_value),
			trainable=True
		)

		self.equalizer = tf.sqrt(2 / input_shape[1])

		super().build(input_shape)

	def call(self, inputs):
		return tf.matmul(inputs, self.equalized_weights * self.equalizer) + self.equalized_biases[None, :]

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'units': self.units,
			'bias_init_value': self.bias_init_value
		})
		return config