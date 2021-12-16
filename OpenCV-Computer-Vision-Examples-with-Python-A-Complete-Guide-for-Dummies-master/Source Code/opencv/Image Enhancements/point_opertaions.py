import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("lena.png")
#=====================================================
# Image Negative
#=====================================================
img_negative = 255 - img

h, w, ch = img.shape

img_neg = np.zeros(img.shape, img.dtype)
for x in range(h):
    for y in range(w):
        for z in range(ch):
            img_neg[x , y, z]=255 - img[x, y, z]

images = np.hstack((img, img_negative, img_neg))
cv2.imshow("Negative of image", images)
cv2.waitKey()
cv2.destroyAllWindows()

#================================================================
#Lograithmic Function
#================================================================
#calculating the value of c
c = 255 / (np.log(1 + np.max(img)))

log_transformed = c * np.log( 1 + img)
log_transformed= np.array(log_transformed, dtype=np.uint8)
log_transformation = np.zeros(img.shape, img.dtype)
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        for z in range(img.shape[2]):
            log_transformation[x, y, z] = c * np.log( 1 + img[x, y, z])
images2 = np.hstack((img, log_transformed, log_transformation))
cv2.imshow("Logarithmic Transformation", images2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#======================================================================
# Power Law Transformation or Gamma Correction
#======================================================================
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
images3=[]
for gamma in [0.1, 0.5, 1.2, 2.2]:
    gamma_corrected = np.array(255*(img/255) ** gamma, dtype="uint8")
    images3.append(gamma_corrected)
images3.append(img)
title=["0.1", "0.5", "1.2", "2.2","Original image"]
for i in range(len(images3)):
    plt.subplot( 3, 3, i+1), plt.imshow(images3[i], cmap="gray")
    plt.title(title[i])
plt.show()
# cv2.imshow("Gamma Correction", gamma_corrected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
