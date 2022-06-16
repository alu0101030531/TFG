from Autoencoder.autoencoder import Autoencoder
from ../Autoencoder/autoencoder import Autoencoder

autoencoder = Autoencoder(100, 100, 3)
autoencoder.createLayers()
autoencoder.readData("hands100x100")
autoencoder.model()
autoencoder.fit()
autoencoder.save("autoencoder")
model = tf.keras.models.load_model("autoencoder")

data = ImageData("hands100x100")
data.readData(10)
data = data.getData()
hands = model.predict(data)#.encoderPredict(data)
#lerp = Lerp(Vector(list(new_predictions[1])), Vector(list(new_predictions[2])), 0.1)
#vectors = lerp.lerp()
#hands = autoencoder.decoderPredict(vectors)
for i in range(len(hands)): 
  img = tf.keras.utils.array_to_img(hands[i])
  img.save(f"interpolation/hand_{i}.png")