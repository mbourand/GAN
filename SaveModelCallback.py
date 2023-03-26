from keras.callbacks import Callback
import os
from settings import *

class SaveModelCallback(Callback):
	def __init__(self):
		super().__init__()

	def on_epoch_end(self, epoch, logs=None):
		if not os.path.exists(MODEL_OUTPUT_PATH):
			os.makedirs(MODEL_OUTPUT_PATH)

		self.model.generator.save(f'{MODEL_OUTPUT_PATH}/generator.h5')
		self.model.discriminator.save(f'{MODEL_OUTPUT_PATH}/discriminator.h5')
