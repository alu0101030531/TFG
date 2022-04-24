from decoder import Decoder
from encoder import Encoder
from data import Data
from keras.callbacks import TensorBoard
import keras

class Autoencoder:
  def __init__(self, width, height, dimensions) -> None:
    self.width = width
    self.height = height
    self.dimensions = dimensions
  
  def createLayers(self):
    self.input_img = keras.Input(shape=(self.width, self.height, self.dimensions))
    self.encoder = Encoder(self.input_img)
    self.encoded = self.encoder.encoderLayers()
    self.decoder = Decoder(self.encoded)
    self.decoded = self.decoder.decoderLayers()

  def readData(self, path):
    self.data = Data(path)
    self.train, self.test = self.data.prepareData()

  def model(self):
    self.autoencoder = keras.Model(self.input_img, self.decoded)
    self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  
  def fit(self):
    self.autoencoder.fit(self.train, self.train,
                epochs=50,
                batch_size=5,
                shuffle=True,
                validation_data=(self.test, self.test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder = Autoencoder(940, 940, 3)
autoencoder.createLayers()
autoencoder.readData("test")
autoencoder.model()
autoencoder.fit()