import cv2
import numpy as np

img= cv2.imread("lena.png")
img = cv2.resize(img, (400,500))
new_image = np.zeros(img.shape, img.dtype)

alpha= 1.5
beta=0
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            new_image[y,x,c] = np.clip(alpha * img[y,x,c] + beta, 0, 255)
new_image2 = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
images= np.hstack((img, new_image, new_image2))
cv2.imshow("Brightness of Image", images)
cv2.waitKey(0)
cv2.destroyAllWindows()