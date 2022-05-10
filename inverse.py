import imageio
import matplotlib.pyplot as plt

img = imageio.imread("image_at_epoch_0085.png")
inverse = []
for i in range(0, len(img)):
  inverse.append([])
  for j in range(0, len(img[i])):
    inverse[i].append([])
    for rgb in range(0, 3):
      inverse[i][j].append((255 + img[i][j][rgb]) % 255)
    inverse[i][j].append(255)
plt.imsave("reverse.png", inverse)
