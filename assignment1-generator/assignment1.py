import numpy as np
import imageio
import matplotlib.pyplot as plt

A = np.zeros(10)
print(A)

filename1 = str(input())
filename2 = str(input())

img1 = imageio.imread(filename1)
img2 = imageio.imread(filename2)

size1 = img1.shape
size2 = img2.shape
assert size1 == size2

imgout = np.zeros(size1, dtype=float)

for x in range(size1[0]):
    for y in range(size1[1]):
        imgout[x, y] = float(img1[x, y]) - float(img2[x, y])


plt.imshow(imgout, cmap='gray')
plt.colorbar()

np.min(imgout)


np.max(imgout)

imax = np.max(imgout)
imin = np.min(imgout)
imgout_norm = (imgout - imin)/(imax-imin)


plt.imshow(imgout_norm, cmap="gray")
plt.colorbar()

imgout_norm = (imgout_norm*255).astype(np.uint8)

plt.imshow(imgout_norm, cmap="gray")
plt.colorbar()

imageio.imwrite("numeros_diff.jpg", imgout_norm)
