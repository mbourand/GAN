from keras.callbacks import Callback
import os
from settings import *
import tensorflow as tf

class SaveModelCallback(Callback):
	def __init__(self):
		super().__init__()

	def on_batch_end(self, batch, logs=None):
		if self.model.step % SAVE_MODEL_FREQUENCY != 0:
			return

		if not os.path.exists(MODEL_OUTPUT_PATH):
			os.makedirs(MODEL_OUTPUT_PATH)
		self.model.save_weights(MODEL_OUTPUT_PATH)
