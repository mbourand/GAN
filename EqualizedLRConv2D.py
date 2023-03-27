from keras.layers import Layer, Conv2D
import tensorflow as tf
from keras.initializers import Constant, GlorotUniform

class EqualizedLRConv2D(Layer):
	def __init__(self, filters, kernel_size, strides=1, padding='same', kernel_initializer=GlorotUniform(), **kwargs):
		super().__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.strides = strides
		self.padding = padding
		self.initializer = kernel_initializer

	def build(self, input_shape):
		self.equalized_weights = self.add_weight(name='equalized_weights', shape=(self.kernel_size[0], self.kernel_size[1], input_shape[3], self.filters), initializer=self.initializer, trainable=True)
		self.equalized_biases = self.add_weight(name='equalized_biases', shape=(self.filters,), initializer=Constant(0), trainable=True)

		self.equalized_weights = self.equalized_weights * tf.sqrt(2 / (self.kernel_size[0] * self.kernel_size[1] * input_shape[3] * self.filters))
		super().build(input_shape)

	def call(self, inputs):
		return tf.nn.conv2d(inputs, self.equalized_weights, strides=(1, self.strides, self.strides, 1), padding=self.padding.upper()) + self.equalized_biases

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1] // self.strides, input_shape[2] // self.strides, self.filters)

	def get_config(self):
		config = super().get_config()
		config['filters'] = self.filters
		config['kernel_size'] = self.kernel_size
		config['strides'] = self.strides
		config['padding'] = self.padding
		return config