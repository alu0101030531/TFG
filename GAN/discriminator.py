from keras.callbacks import TensorBoard
import keras
from keras import layers
from sklearn import metrics
from data import ImageData
import numpy as np
import tensorflow as tf
from generator import Generator
import matplotlib.pyplot as plt

class Discriminator:
  
  def __init__(self, width, height, dimensions) -> None:
      self.inputShape = keras.Input(shape=(width, height, dimensions))
      self.width = width
      self.height = height
      self.dimensions = dimensions
  
  def model(self, filters):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters, 4, strides=(2,2), padding='same', input_shape=[self.width, self.height, self.dimensions]))
    model.add(layers.LeakyReLU(alpha=0.2))  
    model.add(layers.Conv2D(filters * 2, 4, strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(filters * 2, 4, strides=(2, 2), padding='same'))
    model.add()
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


''' 
  def model(self):
    self.discriminatorModel = keras.Model(self.inputShape, self.discriminatorLayers)
    self.discriminatorModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  def readData(self):
    data = ImageData("test")
    batch = 10
    testSize = 0.2
    data.readData(batch)
    self.xtrain, self.xtest = data.prepareData(testSize, True)
    self.ytrain = np.ones(len(self.xtrain))
    self.ytest = np.ones(len(self.xtest))
    fake_xtrain = np.random.uniform(size=((int(batch * (1 - testSize)), self.width,self.height,self.dimensions)))
    fake_xtest = np.random.uniform(size=((int(batch * testSize), self.width,self.height,self.dimensions)))
    self.xtrain = np.concatenate((self.xtrain, fake_xtrain))
    self.xtest = np.concatenate((self.xtest, fake_xtest))
    self.ytrain = np.concatenate((self.ytrain, np.zeros(len(fake_xtrain))))
    self.ytest = np.concatenate((self.ytest, np.zeros(len(fake_xtest))))

  def fit(self):
    self.discriminatorModel.fit(self.xtrain, self.ytrain,
        epochs=20,
        batch_size=4,
        shuffle=True,
        verbose=1,
        validation_data=(self.xtest, self.ytest),
        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

  def save(self, path):
    self.discriminatorModel.save(path)
'''