from keras.layers import Layer
import tensorflow as tf

class MinibatchStandardDeviation(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs):
		mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
		squ_diffs = tf.square(inputs - mean)
		mean_sq_diff = tf.reduce_mean(squ_diffs, axis=0, keepdims=True) + 1e-8
		stdev = tf.sqrt(mean_sq_diff)
		mean_pix = tf.reduce_mean(stdev, keepdims=True)
		shape = tf.shape(inputs)
		output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

		return tf.concat([inputs, output], axis=-1)

	def get_config(self):
		config = super().get_config().copy()
		return config