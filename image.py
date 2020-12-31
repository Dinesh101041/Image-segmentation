import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np

# read images
image = cv2.imread('./images/koala.jpg')
# plot image
plt.imshow(image)
# display the image
plt.show()


# covert image from BGR to RGB
rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgbimage)
plt.show()


# noramlizing 
pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


# visualizing the rgb color
r,g,b = cv2.split(rgbimage)
vis = plt.figure()
axis = vis.add_subplot(1, 1, 1, projection="3d")

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()


# visualizing the hsv color
hsv_nemo = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()