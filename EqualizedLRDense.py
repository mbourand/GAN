from keras.layers import Layer
import tensorflow as tf
from keras.activations import *
from keras.initializers import Constant, RandomNormal
from settings import *

class EqualizedLRDense(Layer):
	def __init__(self, units, kernel_initializer=RandomNormal(stddev=RANDOM_NORMAL_STD_DEV), **kwargs):
		super().__init__(**kwargs)
		self.units = units
		self.initializer = kernel_initializer

	def build(self, input_shape):
		self.equalized_weights = self.add_weight(name='equalized_weights', shape=(input_shape[1], self.units), initializer=self.initializer, trainable=True)
		self.equalized_biases = self.add_weight(name='equalized_biases', shape=(self.units,), initializer=Constant(0), trainable=True)

		self.equalized_weights = self.equalized_weights * tf.sqrt(2 / input_shape[1])
		super().build(input_shape)

	def call(self, inputs):
		return tf.matmul(inputs, self.equalized_weights) + self.equalized_biases

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.units)

	def get_config(self):
		config = super().get_config()
		config['units'] = self.units
		return config