import keras
from data import ImageData, TextData
from lerp import Lerp
from vector import Vector
from matplotlib import pyplot as plt
import numpy

class Precision:
  def __init__(self, encoder, mlp, imgPath, quaternionsPath) -> None:
    self.encoder = encoder
    self.mlp = mlp
    self.imgPath = imgPath
    self.quaternionsPath = quaternionsPath
  
  def predict(self):
    self.data = ImageData(self.imgPath)
    self.data.readData()
    self.data = self.data.getData()
    hands = self.encoder.predict(self.data)
    self.predictedHands = self.mlp.predict(hands)
    joints = TextData(self.quaternionsPath)
    joints.readData()
    self.hands = joints.getData()
    #print(self.predictedHands[0])
    #print(self.hands[0])
    for element in range(0, len(self.predictedHands[1]), 4):
      print(self.predictedHands[1][element], self.predictedHands[1][element+1], self.predictedHands[1][element+2], self.predictedHands[1][element+3])
  def calculatePrecision(self):
    badPredictions = 0
    for joint, predictedJoint in zip(self.hands, self.predictedHands):
      predicted = True
      sumVal1 = 0
      sumVal2 = 0
      for val1, val2 in zip(joint, predictedJoint):
        sumVal1 += val1
        sumVal2 += val2
      if abs(sumVal1 - sumVal2) > 0.4:
        badPredictions += 1
    return (len(self.hands) - badPredictions) / len(self.hands)

precision = Precision(keras.models.load_model("encoder"), keras.models.load_model("mlp"), "test", "quaternions.txt")
precision.predict()
print(precision.calculatePrecision())

'''
  encoder = keras.models.load_model("encoder")
  model = keras.models.load_model("mlp")
  data = ImageData("test")
  data.readData()
  train, test = data.splitData(0.2, False)
  #new_predictions 
  hands = encoder.predict(test)
  joints = model.predict(hands)
  print(joints[0])
  #lerp = Lerp(Vector(list(new_predictions[0])), Vector(list(new_predictions[1])), 0.1)
  #vectors = lerp.lerp()
  #decoder = keras.models.load_model("decoder.d")
  #hands = decoder.predict(vectors)
'''