from keras.models import Model
from keras.losses import *
from keras.optimizers import *
from mapping_model import get_mapping_model
from generator_model import get_generator_model
from discriminator_model import get_discriminator_model

from settings import *

import tensorflow as tf
from keras.utils import plot_model

import os

class GANModel(Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.mapper = get_mapping_model()
		self.generator = get_generator_model()
		plot_model(self.generator, to_file='generator.png', show_shapes=True, show_layer_names=False)
		self.discriminator = get_discriminator_model()
		plot_model(self.discriminator, to_file='discriminator.png', show_shapes=True, show_layer_names=False)

		self.step = tf.Variable(0, name='step', trainable=False, dtype=tf.int32)

	def compile(self, **kwargs):
		self.generator_optimizer = Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2, epsilon=ADAM_EPSILON)
		self.discriminator_optimizer = Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2, epsilon=ADAM_EPSILON)

		super().compile(**kwargs)

	def get_w(self, batch_size):
		rand = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)

		if rand < STYLE_MIXING_PROBABILITY:
			z = tf.random.normal((batch_size, LATENT_DIM))
			z2 = tf.random.normal((batch_size, LATENT_DIM))
			w = self.mapper(z, training=True)
			w2 = self.mapper(z2, training=True)

			switch_block_index = tf.random.uniform(shape=(), minval=1, maxval=BLOCKS_COUNT, dtype=tf.int32)

			return [w if i < switch_block_index else w2 for i in range(BLOCKS_COUNT)]
		else:
			z = tf.random.normal((batch_size, LATENT_DIM))
			w = self.mapper(z, training=True)
			return [w for _ in range(BLOCKS_COUNT)]

	def get_noises(self, batch_size):
		return [tf.random.normal((batch_size, IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range(BLOCKS_COUNT * 2 + 1)]

	def generator_loss(self, fake_output):
		return tf.reduce_mean(tf.nn.softplus(-fake_output))

	def discriminator_loss(self, real_output, fake_output):
		return tf.reduce_mean(tf.nn.softplus(fake_output) + tf.nn.softplus(-real_output))

	def gradient_penalty(self, real_output, real_images):
		gradients = tf.gradients(tf.reduce_sum(real_output), [real_images])[0]
		gradient_penalty = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])

		return tf.reduce_mean(gradient_penalty) * R1_GAMMA * 0.5 * PENALTY_FREQUENCY

	@tf.function
	def train_step(self, data):
		batch_size = tf.shape(data)[0]

		gradient_penalty = 0.0
		with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
			latents = self.get_w(batch_size)
			noises = self.get_noises(batch_size)
			fake_images = self.generator([tf.ones((batch_size, 1))] + latents + noises, training=True)

			real_output = self.discriminator(data, training=True)
			fake_output = self.discriminator(fake_images, training=True)

			generator_loss = self.generator_loss(fake_output)
			discriminator_loss = self.discriminator_loss(real_output, fake_output)

			if self.step % PENALTY_FREQUENCY == 0:
				gradient_penalty = self.gradient_penalty(real_output, data)

			generator_gradient = generator_tape.gradient(generator_loss, self.generator.trainable_variables + self.mapper.trainable_variables)
			discriminator_gradient = discriminator_tape.gradient(discriminator_loss + gradient_penalty, self.discriminator.trainable_variables)

			self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables + self.mapper.trainable_variables))
			self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

		self.step.assign_add(1)

		return { 'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss }

	def predict(self, x, **kwargs):
		latents = self.get_w(x.shape[0])
		noises = self.get_noises(x.shape[0])
		return self.generator([tf.ones((x.shape[0], 1))] + latents + noises, training=False)

	def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
		self.mapper.save_weights(os.path.join(filepath, 'mapper.h5'), overwrite, save_format, options)
		self.generator.save_weights(os.path.join(filepath, 'generator.h5'), overwrite, save_format, options)
		self.discriminator.save_weights(os.path.join(filepath, 'discriminator.h5'), overwrite, save_format, options)
		tf.saved_model.save(self.step, os.path.join(filepath, 'step.h5'))

	def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
		self.mapper.load_weights(os.path.join(filepath, 'mapper.h5'), by_name, skip_mismatch, options)
		self.generator.load_weights(os.path.join(filepath, 'generator.h5'), by_name, skip_mismatch, options)
		self.discriminator.load_weights(os.path.join(filepath, 'discriminator.h5'), by_name, skip_mismatch, options)
		self.step = tf.saved_model.load(self.step, os.path.join(filepath, 'step.h5'))