import cv2
import matplotlib.pyplot as plt
import numpy as np
img= cv2.imread("img.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imshow("Original Image",img)
# cv2.waitKey(0)
rows,cols,chns=img.shape
M=np.float32([[1, 0, 50],
             [0, 1, 50],
             [0, 0, 1]])
translated_image= cv2.warpPerspective(img, M, (cols,rows))
plt.axis('off')
# show the resulting image
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(translated_image)
plt.show()