from keras.callbacks import TensorBoard
import keras
from keras import layers
import numpy as np
from urllib3 import Retry
import tensorflow as tf
import matplotlib.pyplot as plt

class Generator:
  def __init__(self, width, height, dimensions) -> None:
      self.width = width
      self.height = height
      self.dimensions = dimensions

  def model(self, latent_dimension, size):
      model = tf.keras.Sequential()
      model.add(layers.Dense(int(size / 8 * size / 8 * latent_dimension), input_shape=(latent_dimension,)))
      model.add(layers.Reshape((int(size / 8), int(size / 8), latent_dimension)))
      model.add(layers.Conv2DTranspose(latent_dimension, 4, strides=2, padding='same'))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.Conv2DTranspose(latent_dimension * 2, 4, strides=2, padding='same'))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.Conv2DTranspose(latent_dimension * 4, 4, strides=2, padding='same'))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.Conv2D(3, 5, activation='sigmoid', padding='same'))
      return model


