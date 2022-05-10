import os

def rename(path):
  name = 6006
  for filename in os.listdir(path):
    os.rename(path + "/" + filename, "test/" + str(name) + ".png")
    name += 1

rename("test2")