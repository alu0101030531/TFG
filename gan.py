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
    self.generator = Generator(940, 940, 32).model(32)
    self.discriminator = Discriminator(940, 940, 3).model()
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
    base_epoch = 165 + epoch
    predictions = self.generator(self.seed, training=False)
    tf.keras.utils.save_img('img2/image_at_epoch_{:04d}.png'.format(base_epoch), predictions[0] * 255)

  def restore(self, path):
    self.checkpoint.restore(tf.train.latest_checkpoint(path))

gan = Gan(940,940,3)
images = ImageData('test')
images.readData(100)
train_images = images.normalizeImages()
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(5)
gan.train(train_dataset, 500)