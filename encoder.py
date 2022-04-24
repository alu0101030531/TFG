import keras
from keras import layers

class Encoder:
  def __init__(self, input_img) -> None:
      self.input_img = input_img

  def encoderLayers(self):
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    self.encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    print(self.encoded.shape)
    return self.encoded
  