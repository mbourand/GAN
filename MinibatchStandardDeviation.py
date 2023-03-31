from keras.layers import Layer
import tensorflow as tf

class MinibatchStandardDeviation(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs):
		square_diffs = tf.square(inputs - tf.reduce_mean(inputs, axis=0, keepdims=True))
		mean_square_diff = tf.reduce_mean(square_diffs, axis=0, keepdims=True)
		stddev = tf.sqrt(mean_square_diff)

		mean_stddev = tf.reduce_mean(stddev, keepdims=True)

		shape = tf.shape(inputs)
		output = tf.tile(mean_stddev, [shape[0], shape[1], shape[2], 1])

		return tf.concat([inputs, output], axis=-1)

	def get_config(self):
		config = super().get_config().copy()
		return config