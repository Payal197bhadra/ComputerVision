import cv2
import matplotlib.pyplot as plt
import numpy as np
img= cv2.imread("img.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# get the image shape
rows, cols, dim = img.shape
#angle from degree to radian
angle = np.radians(10)
#transformation matrix for Rotation
M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	[np.sin(angle), np.cos(angle), 0],
            	[0, 0, 1]])
# apply a perspective transformation to the image
rotated_img = cv2.warpPerspective(img, M, (int(cols),int(rows)))
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(rotated_img)
plt.show()
