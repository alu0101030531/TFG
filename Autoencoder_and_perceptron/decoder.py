from quopri import encodestring
from keras import layers
import keras


class Decoder:
  def __init__(self, encoder) -> None:
    self.encoder = encoder
    self.input_img = keras.Input(shape=(13,13,8))

  def decoderLayers(self):
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.input_img)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    #x = layers.Conv2D(16, (3,3), activation='relu')(x)
    #x = layers.UpSampling2D((2, 2))(x)
    self.decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return self.decoded
  
  def model(self):
    self.decoder = keras.Model(self.input_img, self.decoded)
    return self.decoder
  
  def save(self, path):
    self.decoder.compile(optimizer='adam', loss='binary_crossentropy')
    self.decoder.save(path)
