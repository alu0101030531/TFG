import keras
from data import ImageData, TextData
from lerp import Lerp
from vector import Vector
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy

class Precision:
  def __init__(self, encoder, mlp, imgPath, quaternionsPath) -> None:
    self.encoder = encoder
    self.mlp = mlp
    self.imgPath = imgPath
    self.quaternionsPath = quaternionsPath
  
  def predict(self):
    self.data = ImageData(self.imgPath)
    self.data.readData(1000)
    self.data = self.data.getData()
    hands = self.encoder.predict(self.data)
    self.predictedHands = self.mlp.predict(hands)
    joints = TextData(self.quaternionsPath)
    joints.readData(1000)
    self.hands = joints.getData()
    for element in range(0, len(self.predictedHands[1]), 4):
      print(self.predictedHands[1][element], self.predictedHands[1][element+1], self.predictedHands[1][element+2], self.predictedHands[1][element+3])

  def calculatePrecision(self):
    badPredictions = 0
    for joint, predictedJoint in zip(self.hands, self.predictedHands):
      for val1, val2 in zip(joint, predictedJoint):
        if abs(val1 - val2) > 0.01:
          badPredictions += 1
        break
#      if abs(abs(sumVal1) - abs(sumVal2)) > 1:
#        badPredictions += 1
    return (len(self.hands) - badPredictions) / len(self.hands)

precision = Precision(keras.models.load_model("encoder"), keras.models.load_model("mlp"), "Datasets/TestHands64x64", "Quaternions/TestHands64x64quaternions.txt")
precision.predict()
print(precision.calculatePrecision())
