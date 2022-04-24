from vector import Vector

class Lerp:
  def __init__(self, vector1, vector2, factor):
    self.vector1 = vector1
    self.vector2 = vector2
    self.factor = factor
  
  def __str__(self):
    return str(self.vector1) + "\n" + str(self.vector2)

  def lerp(self):
    vectors = []
    factorValue = 0
    while(factorValue <= 1):
      a = self.lerpStep(factorValue)
      print(a.vector)
      vectors.append(a)
      factorValue = round(factorValue + self.factor, 1)
    return vectors

  def lerpStep(self, factor):
    a = self.vector1.scalarMultiplication(1 - factor)
    b = self.vector2.scalarMultiplication(factor)
    return a.addition(b)

lerp = Lerp(Vector([[[2, 3, 6]]]), Vector([[[1, 2, 3]]]), 0.1)
lerp.lerp()

