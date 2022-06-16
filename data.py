from abc import abstractmethod
from importlib.resources import path
import os
from sklearn.model_selection import train_test_split
import numpy as np
import imageio
from PIL import Image
from urllib3 import Retry
from random import randint

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class Data:
  def __init__(self, path) -> None:
    self.path = path
    self.data = []
    #self.readData()
  
  @abstractmethod
  def readData(self):
    pass

  def splitData(self, percentage, shuffle):
    self.data_train, self.data_test = train_test_split(self.data, test_size= percentage, shuffle=shuffle)
    return np.array(self.data_train), np.array(self.data_test)
  
  def normalizeData(self, maxValue):
    x_train = np.array(self.data_train).astype('float32') / maxValue
    x_test = np.array(self.data_test).astype('float32') / maxValue
    return x_train, x_test
  
  def normalizeImages(self):
    return np.array(self.data).astype('float32') / 255

  def prepareData(self, testSize, shuffle):
    self.splitData(testSize, shuffle)
    return self.normalizeData(255)
  
  def getData(self):
    return np.array(self.data)

class ImageData(Data):
  def readData(self, samples):
    images = os.listdir(self.path)
    images = sorted_alphanumeric(images)
    for filename in range(0, samples):
      img = np.asarray(Image.open(self.path + "/" + images[filename]))
      #img = img.reshape(940, 940, 1)
      self.data.append(img)

'''len(lines)'''
class TextData(Data):
  def readData(self, samples):
    file = open(self.path, 'r')
    lines = file.readlines()
    data = []
    iterator = 0
    for line in range(0, samples):
      data.append([])
      for joints in range(0, len(lines[line].split(" ")[:-1])):
        for coords in range(0, len(lines[line].split(" ")[:-1][joints].split(":"))):
          data[iterator].append(float(lines[line].split(" ")[:-1][joints].split(":")[coords].replace(',', '.')))
      iterator += 1
    self.data = data