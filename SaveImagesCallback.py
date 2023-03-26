from keras.callbacks import Callback
from PIL import Image
from settings import *
import tensorflow as tf
import numpy as np
import os

class SaveImagesCallback(Callback):
	def __init__(self):
		super().__init__()
		self.count = 0
		self.latent = tf.random.normal((OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1], LATENT_DIM), seed=SEED)

	def save_images(self, images):
		output_image = np.full((
			MARGIN + (OUTPUT_SHAPE[1] * (images.shape[2] + MARGIN)),
			MARGIN + (OUTPUT_SHAPE[0] * (images.shape[1] + MARGIN)),
			images.shape[3]), 255, dtype = np.uint8
		)

		i = 0
		for row in range(OUTPUT_SHAPE[1]):
			for col in range(OUTPUT_SHAPE[0]):
				r = row * (images.shape[2] + MARGIN) + MARGIN
				c = col * (images.shape[1] + MARGIN) + MARGIN
				output_image[r:r + images.shape[2], c:c + images.shape[1]] = images[i]
				i += 1

		if not os.path.exists(IMAGE_OUTPUT_PATH):
			os.makedirs(IMAGE_OUTPUT_PATH)

		img = Image.fromarray(output_image)
		img.save(os.path.join(IMAGE_OUTPUT_PATH, f'image_{self.count // SAVE_IMAGE_FREQUENCY}.png'))
		img.save(os.path.join(IMAGE_OUTPUT_PATH, 'last_image.png'))

	def on_batch_end(self, batch, logs=None):
		images = (self.model.predict(self.latent) + 1) * 127.5
		if self.count % SAVE_IMAGE_FREQUENCY == 0:
			self.save_images(images)
		self.count += 1
