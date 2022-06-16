from decoder import Decoder
from encoder import Encoder
from data import Data, ImageData
from keras.callbacks import TensorBoard
from tensorflow import keras
import tensorflow as tf

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
    self.data.readData(500)
    self.train, self.test = self.data.prepareData(0.2, True)

  def predict(self):
    predictions = self.autoencoder.predict(self.test)
    for i in range(10):
      predicted_img = tf.keras.utils.array_to_img(predictions[i])
      predicted_img.save(f"predict2/predict_940_{i}.png")
      real_img = tf.keras.utils.array_to_img(self.test[i])
      real_img.save(f"predict2/real_940_{i}.png")

  def encoderPredict(self, imgs):
    return self.encoderModel.predict(imgs)

  def decoderPredict(self, imgs):
    return self.decoderModel.predict(imgs)

  def model(self):
    self.encoderModel = self.encoder.model()
    self.decoderModel = self.decoder.model()
    self.encoded = self.encoderModel(self.input_img)
    self.decoded = self.decoderModel(self.encoded)
    self.autoencoder = keras.Model(self.input_img, self.decoded)
    self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  
  def fit(self):
    self.autoencoder.fit(self.train, self.train,
                epochs=50,
                batch_size=10,
                shuffle=True,
                verbose=1,
                validation_data=(self.test, self.test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
  
  def save(self, path):
    tf.keras.models.save_model(self.autoencoder, path, save_format="tf")
  
  def saveEncoder(self, path):
    self.encoder.model()
    self.encoder.save_model(path, save_format="tf")
  
  def saveDecoder(self, path):
    self.decoder.model()
    self.decoder.save_model(path, save_format="tf")

autoencoder = Autoencoder(100, 100, 3)
autoencoder.createLayers()
autoencoder.readData("hands100x100")
autoencoder.model()
autoencoder.fit()
autoencoder.save("autoencoder")
autoencoder.saveEncoder("encoder")
autoencoder.saveDecoder("decoder")
autoencoder.predict()
