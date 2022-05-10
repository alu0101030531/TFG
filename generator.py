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

  def model(self, dimension):
    model = tf.keras.Sequential()
    model.add(layers.Dense(118 * 118 * 32, input_shape=(dimension,)))
    model.add(layers.Reshape((118, 118, 32)))
    model.add(layers.Conv2DTranspose(16, 4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(16, 4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(16, 4, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(3, 5, activation='sigmoid'))
    return model


