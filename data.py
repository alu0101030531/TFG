from importlib.resources import path
import os
from sklearn.model_selection import train_test_split
import numpy as np
import imageio

class Data:
  def __init__(self, path) -> None:
    self.path = path
    self.readData()
  
  def readData(self):
    self.data = []
    for filename in os.listdir(self.path):
      img=imageio.imread(self.path + "/" + filename)
      self.data.append(img)

  def splitData(self, percentage):
    self.data_train, self.data_test = train_test_split(self.data, test_size= percentage, shuffle=True)
  
  def normalizeData(self, maxValue):
    x_train = np.array(self.data_train).astype('float32') / maxValue
    x_test = np.array(self.data_test).astype('float32') / maxValue
    return x_train, x_test
  
  def prepareData(self):
    self.splitData(0.2)
    return self.normalizeData(255)

