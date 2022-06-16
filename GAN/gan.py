from matplotlib import image
from pyrfc3339 import generate
from discriminator import Discriminator
from generator import Generator
from data import ImageData
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os

class Gan:
  def __init__(self, width, height, dimensions) -> None:
    self.width = width
    self.height = height
    self.dimensions = dimensions
    self.crossEntropy = tf.keras.losses.BinaryCrossentropy()
    self.generatorOptimizer = tf.keras.optimizers.Adam(1e-4)
    self.discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)
    self.noiseDim = 32
    self.examplesToGenerate = 16
    self.seed = tf.random.normal([self.examplesToGenerate, self.noiseDim])
    self.generator = Generator(100, 100, 32).model(32)
    self.discriminator = Discriminator(100, 100, 3).model()
    self.checkpoint('./RGBtraining_checkpoints')

  def checkpoint(self, path):
    self.checkpointPrefix = os.path.join(path, "ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generatorOptimizer,
                                 discriminator_optimizer=self.discriminatorOptimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)

  def discriminatorLoss(self, realOutput, fakeOutput):
    real_loss = self.crossEntropy(tf.ones_like(realOutput), realOutput)
    fake_loss = self.crossEntropy(tf.zeros_like(fakeOutput), fakeOutput)
    total_loss = real_loss + fake_loss
    return total_loss
  
  def generatorLoss(self, fakeOutput):
    return self.crossEntropy(tf.ones_like(fakeOutput), fakeOutput)

  def trainStep(self, images, batchSize):
    noise = tf.random.uniform([batchSize, self.noiseDim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generatedImages = self.generator(noise, training=True)
      realOutput = self.discriminator(images, training=True)
      fakeOutput = self.discriminator(generatedImages, training=True)

      genLoss = self.generatorLoss(fakeOutput)
      discLoss = self.discriminatorLoss(realOutput, fakeOutput)

      gradientsOfGenerator = gen_tape.gradient(genLoss, self.generator.trainable_variables)
      gradientsOfDiscriminator = disc_tape.gradient(discLoss, self.discriminator.trainable_variables)

      self.generatorOptimizer.apply_gradients(zip(gradientsOfGenerator, self.generator.trainable_variables))
      self.discriminatorOptimizer.apply_gradients(zip(gradientsOfDiscriminator, self.discriminator.trainable_variables))
  
  def train(self, dataset, epochs):
    for epoch in range(epochs):
      start = time.time()

      for imageBatch in dataset:
        self.trainStep(imageBatch, 5)
      
      self.generateAndSaveImages(epoch + 1)
      if (epoch + 1) % 15 == 0:
        self.checkpoint.save(self.checkpointPrefix)

      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    self.generateAndSaveImages(epoch + 1)

  def generateAndSaveImages(self, epoch):
    base_epoch = 0 + epoch
    predictions = self.generator(self.seed, training=False)
    tf.keras.utils.save_img('img2/image_at_epoch_{:04d}.png'.format(base_epoch), predictions[0] * 255)

  def restore(self, path):
    self.checkpoint.restore(tf.train.latest_checkpoint(path))


class GAN(keras.Model):
  def __init__(self, size, channels, filters, latentDimension):
      super().__init__()
      self.generator = Generator(size, size, latentDimension).model(latentDimension, size)
      self.discriminator = Discriminator(size, size, channels).model(filters)
      self.latent_dim = latentDimension
      self.d_loss_metric = keras.metrics.Mean(name="d_loss")
      self.g_loss_metric = keras.metrics.Mean(name="g_loss")
  
  def compile(self, d_optimizer, g_optimizer, loss_fn):
    super(GAN, self).compile()
    self.d_optimizer = d_optimizer
    self.g_optimizer = g_optimizer
    self.loss_fn = loss_fn
  
  @property
  def metrics(self):
    return [self.d_loss_metric, self.g_loss_metric]
  
  def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]
    random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    generated_images = self.generator(random_latent_vectors)
    combined_images = tf.concat([generated_images, real_images], axis=0)
    labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],axis=0)

    labels += 0.05 * tf.random.uniform(tf.shape(labels))

    with tf.GradientTape() as tape:
      predictions = self.discriminator(combined_images)
      d_loss = self.loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

    random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    misleading_labels = tf.zeros((batch_size, 1))

    with tf.GradientTape() as tape:
      predictions = self.discriminator(self.generator(random_latent_vectors))
      g_loss = self.loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, self.generator.trainable_weights)
    self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

    self.d_loss_metric.update_state(d_loss)
    self.g_loss_metric.update_state(g_loss)
    return {"d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()}

class GANMonitor(keras.callbacks.Callback):
  def __init__(self, num_img=3, latent_dim=100):
    self.num_img = num_img
    self.latent_dim = latent_dim

  def on_epoch_end(self, epoch, logs=None):
    random_latent_vectors = tf.random.normal(
    shape=(self.num_img, self.latent_dim))
    generated_images = self.model.generator(random_latent_vectors)
    generated_images *= 255 
    generated_images.numpy()
    for i in range(self.num_img):
      img = tf.keras.utils.array_to_img(generated_images[i])
      img.save(f"Generated/generated64x64/generated_img_{epoch:03d}_{i}.png")

EPOCHS = 100
FILTERS = 32
LATENT_DIM = FILTERS * 2
gan = GAN(64, 3, FILTERS, LATENT_DIM)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)


train_dataset = tf.keras.preprocessing.image_dataset_from_directory("Datasets/hands64x64", label_mode=None, batch_size=32, image_size=(64,64))
train_dataset = train_dataset.map(lambda x: x / 255).take(300)

gan.fit(
    train_dataset, epochs=EPOCHS,
    callbacks=[GANMonitor(num_img=1, latent_dim=LATENT_DIM)]
)

'''
gan = Gan(100,100,3)
images = ImageData('test')
images.readData(500)
train_images = images.normalizeImages()
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(5)
gan.train(train_dataset, 500)
'''