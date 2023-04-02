from keras.layers import Layer
from keras.initializers import Constant

class BiasLayer(Layer):
	def __init__(self, bias_init_value=0, **kwargs):
		super().__init__(**kwargs)
		self.bias_init_value = bias_init_value

	def build(self, input_shape):
		self.bias = self.add_weight(
			name='bias',
			shape=(input_shape[-1],),
			initializer=Constant(self.bias_init_value),
			trainable=True
		)

		super().build(input_shape)

	def call(self, inputs):
		return inputs + self.bias[None, None, None, :]

	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'bias_init_value': self.bias_init_value
		})
		return config