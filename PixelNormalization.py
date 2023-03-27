from keras.layers import Layer
from keras import backend as K
import tensorflow as tf

class PixelNormalization(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs):
		return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + 1e-8)
 
	def compute_output_shape(self, input_shape):
		return input_shape
