from keras.models import Model
from keras.losses import *
from keras.optimizers import *
from generator_model import get_generator_model
from discriminator_model import get_discriminator_model
import tensorflow as tf

from settings import *

class GANModel(Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.generator = get_generator_model()
		self.discriminator = get_discriminator_model()

	def compile(self, **kwargs):
		self.generator_optimizer = Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA1)
		self.discriminator_optimizer = Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA1)

		self.gan_loss = BinaryCrossentropy()

		super().compile(**kwargs)

	def generator_loss(self, fake_output):
		return self.gan_loss(tf.ones_like(fake_output), fake_output)

	def discriminator_loss(self, real_output, fake_output):
		real_loss = self.gan_loss(tf.ones_like(real_output) * (1 - SMOOTH), real_output)
		fake_loss = self.gan_loss(tf.zeros_like(fake_output), fake_output)
		return real_loss + fake_loss

	@tf.function
	def train_step(self, data):
		batch_size = tf.shape(data)[0]
		latent = tf.random.normal((batch_size, LATENT_DIM))

		with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
			fake_images = self.generator(latent, training=True)
			real_output = self.discriminator(data, training=True)
			fake_output = self.discriminator(fake_images, training=True)

			generator_loss = self.generator_loss(fake_output)
			discriminator_loss = self.discriminator_loss(real_output, fake_output)

			generator_gradient = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
			discriminator_gradient = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

			self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))
			self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

		return { 'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss }

	def predict(self, x, **kwargs):
		return self.generator(x, training=False)

	def call(self, inputs, training=None, mask=None):
		return super().call(inputs, training, mask)