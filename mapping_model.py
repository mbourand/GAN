from settings import *
from keras.models import Sequential
from keras.layers import *
from EqDense import EqDense
from PixelNormalization import PixelNormalization

def get_mapping_model():
	mapping = Sequential()
	mapping.add(Input(shape=(LATENT_DIM,)))
	mapping.add(PixelNormalization())
	for _ in range(MAPPING_LAYERS):
		mapping.add(EqDense(LATENT_DIM, learning_rate_scale=0.01))
		mapping.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA)),

	return mapping