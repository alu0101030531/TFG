from quopri import encodestring
from keras import layers


class Decoder:
  def __init__(self, encoder) -> None:
    self.encoder = encoder

  def decoderLayers(self):
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.encoder)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    self.decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return self.decoded
