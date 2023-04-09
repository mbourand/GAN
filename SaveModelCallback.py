from keras.callbacks import Callback
import os
from settings import *

class SaveModelCallback(Callback):
	def __init__(self):
		super().__init__()

	def on_epoch_end(self, epoch, logs=None):
		if not os.path.exists(MODEL_OUTPUT_PATH):
			os.makedirs(MODEL_OUTPUT_PATH)

		self.model.save_weights(MODEL_OUTPUT_PATH)
