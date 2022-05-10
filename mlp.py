import keras
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
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    self.mlpLayers = layers.Dense(76, activation='linear')(x)
  
  def readData(self, path, path2):
    encoder = keras.models.load_model("encoder")
    self.data1 = ImageData(path)
    self.data1.readData()
    self.train, self.test = self.data1.prepareData(0.2, False)
    self.train = encoder.predict(self.train)
    self.test = encoder.predict(self.test)
    self.data2 = TextData(path2)
    self.data2.readData()
    self.train2, self.test2 = self.data2.splitData(0.2, False)

  def model(self):
    self.mlpModel = keras.Model(self.inputShape, self.mlpLayers)
    self.mlpModel.compile(optimizer='adam', loss='mse')

  def fit(self):
    self.mlpModel.fit(self.train, self.train2,
          epochs=20,
          batch_size=10,
          shuffle=True,
          verbose=1,
          validation_data=(self.test, self.test2),
          callbacks=[TensorBoard(log_dir='/tmp/mlp')])

  def save(self, path):
    self.mlpModel.save(path)

mlp = Mlp(keras.Input(shape=(59,59,76)))
mlp.createLayers()
mlp.readData("test", "quaternions.txt")
mlp.model()
mlp.fit()
mlp.save("mlp")