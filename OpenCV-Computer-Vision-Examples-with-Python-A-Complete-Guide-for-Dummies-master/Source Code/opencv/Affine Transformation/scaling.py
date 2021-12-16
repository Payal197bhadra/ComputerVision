import cv2
import matplotlib.pyplot as plt
import numpy as np
img= cv2.imread("img.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# get the image shape
rows, cols, dim = img.shape

#transformation matrix for Scaling
M = np.float32([[0.5, 0  , 0],
            	[0,   2, 0],
            	[0,   0,   1]])

# apply a perspective transformation to the image
scaled_img = cv2.warpPerspective(img,M,(cols*3,rows*3))
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(scaled_img)
plt.show()