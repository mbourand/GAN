from keras.layers import Layer
from keras.initializers import Constant

class NoiseLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def build(self, input_shape):
		self.noise_scale = self.add_weight(
			name='noise_scale',
			shape=(1,),
			initializer=Constant(0),
			trainable=True
		)
		super().build(input_shape)

	def call(self, inputs):
		image, noise = inputs
		return image + noise * self.noise_scale

	def get_config(self):
		config = super().get_config().copy()
		return config
