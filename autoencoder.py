from decoder import Decoder
from encoder import Encoder
from data import Data, ImageData
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
    self.data = ImageData(path)
    self.data.readData()
    self.train, self.test = self.data.prepareData(0.2, False)

  def model(self):
    self.encoderModel = self.encoder.model()
    self.decoderModel = self.decoder.model()
    self.encoded = self.encoderModel(self.input_img)
    self.decoded = self.decoderModel(self.encoded)
    self.autoencoder = keras.Model(self.input_img, self.decoded)
    self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  
  def fit(self):
    self.autoencoder.fit(self.train, self.train,
                epochs=20,
                batch_size=4,
                shuffle=True,
                verbose=1,
                validation_data=(self.test, self.test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
  
  def save(self, path):
    self.autoencoder.save(path)
  
  def saveEncoder(self, path):
    #self.encoder.model()
    self.encoder.save(path)
  
  def saveDecoder(self, path):
    #self.decoder.model()
    self.decoder.save(path)

autoencoder = Autoencoder(940, 940, 3)
autoencoder.createLayers()
autoencoder.readData("test")
autoencoder.model()
autoencoder.fit()
autoencoder.save("autoencoder")
autoencoder.saveEncoder("encoder")
autoencoder.saveDecoder("decoder")