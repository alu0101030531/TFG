import keras
import tensorflow as tf
from keras import layers
from pytz import timezone
from data import *
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

class Mlp:
  def __init__(self, inputShape) -> None:
    self.inputShape = inputShape
  
  def createLayers(self):
    x = layers.Flatten()(self.inputShape)
    x = layers.Dense(64, activation="tanh")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(64, activation="tanh")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(128, activation="tanh")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Dropout(0.4)(x)
    x = layers.Dense(80, activation="tanh")(x)
    self.mlpLayers = layers.LeakyReLU(alpha=0.2)(x)
  
  def readData(self, path, path2):
    encoder = keras.models.load_model("encoder")
    self.data1 = ImageData(path)
    self.data1.readData(1000)
    self.train, self.test = self.data1.prepareData(0.2, False)
    self.train = encoder.predict(self.train)
    self.test = encoder.predict(self.test)
    self.data2 = TextData(path2)
    self.data2.readData(1000)
    self.train2, self.test2 = self.data2.splitData(0.2, False)

  def model(self):
    self.mlpModel = keras.Model(self.inputShape, self.mlpLayers)
    self.mlpModel.compile(optimizer='adam', loss='mse')

  def fit(self):
    self.mlpModel.fit(self.train, self.train2,
          epochs=30,
          batch_size=512,
          shuffle=True,
          verbose=1,
          validation_data=(self.test, self.test2),
          callbacks=[TensorBoard(log_dir='/tmp/mlp')])
    

  def save(self, path):
    self.mlpModel.save(path)
    return self.mlpModel

mlp = Mlp(keras.Input(shape=(4,4,8)))
mlp.createLayers()
mlp.readData("Datasets/wristHands64x64", "Quaternions/WristHands64x64quaternions.txt")
mlp.model()
mlp.fit()
mlp = mlp.save("mlp")
