from keras.models import Model
from keras.losses import *
from keras.optimizers import *
from mapping_model import get_mapping_model
from generator_model import get_generator_model
from discriminator_model import get_discriminator_model

from settings import *

import tensorflow as tf
from keras.utils import plot_model

class GANModel(Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.mapper = get_mapping_model()
		self.generator = get_generator_model()
		plot_model(self.generator, to_file='generator.png', show_shapes=True, show_layer_names=False)
		self.discriminator = get_discriminator_model()
		plot_model(self.discriminator, to_file='discriminator.png', show_shapes=True, show_layer_names=False)

	def compile(self, **kwargs):
		self.generator_optimizer = Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2, epsilon=ADAM_EPSILON)
		self.discriminator_optimizer = Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2, epsilon=ADAM_EPSILON)

		super().compile(**kwargs)

	def generator_loss(self, fake_output):
		return tf.reduce_mean(tf.nn.softplus(-fake_output))

	def discriminator_loss(self, real_output, fake_output):
		return tf.reduce_mean(tf.nn.softplus(fake_output) + tf.nn.softplus(-real_output))

	@tf.function
	def train_step(self, data):
		batch_size = tf.shape(data)[0]

		with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
			z = tf.random.normal((batch_size, LATENT_DIM))
			latent = self.mapper(z, training=True)
			fake_images = self.generator([tf.ones((batch_size, 1)), latent], training=True)

			real_output = self.discriminator(data, training=True)
			fake_output = self.discriminator(fake_images, training=True)

			generator_loss = self.generator_loss(fake_output)
			discriminator_loss = self.discriminator_loss(real_output, fake_output)

			generator_gradient = generator_tape.gradient(generator_loss, self.generator.trainable_variables + self.mapper.trainable_variables)
			discriminator_gradient = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

			self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables + self.mapper.trainable_variables))
			self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

		return { 'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss }

	def predict(self, x, **kwargs):
		latent = self.mapper(x, training=False)
		return self.generator([tf.ones((x.shape[0], 1)), latent], training=False)