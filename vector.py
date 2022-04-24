
class Vector:
  def __init__(self, vector):
    self.vector = vector
  
  def scalarMultiplication(self, scalar):
    vector = []
    for pixelh in range(len(self.vector)):
      vector.append([])
      for pixelw in range(len(self.vector[pixelh])):
        vector[pixelh].append([])
        for rgb in range(len(self.vector[pixelh][pixelw])):
          vector[pixelh][pixelw].append(round(self.vector[pixelh][pixelw][rgb] * scalar, 1 ))
    return Vector(vector)
  
  def getVector(self):
    return self.vector

  def addition(self, vector):
    additionVector = []
    for pixelh in range(len(self.vector)):
      additionVector.append([])
      for pixelw in range(len(self.vector[pixelh])):
        additionVector[pixelh].append([])
        for rgb in range(len(self.vector[pixelh][pixelw])):
          additionVector[pixelh][pixelw].append(round(self.vector[pixelh][pixelw][rgb] + vector.getVector()[pixelh][pixelw][rgb], 1))
    return Vector(additionVector)