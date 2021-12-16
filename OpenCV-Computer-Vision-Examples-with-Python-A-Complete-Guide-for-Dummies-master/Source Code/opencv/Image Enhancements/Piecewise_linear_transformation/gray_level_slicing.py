import numpy as np
import cv2
img = cv2.imread("lena.png",0)
row, column= img.shape

#create an zero array yo store the sliced image
img1 = np.zeros((row, column), dtype='uint8')

# Specify the min and max range
min_range = 50
max_range = 80

# Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
for i in range(row):
    for j in range(column):
        if img[i, j] > min_range and img[i, j] < max_range:
            img1[i, j] = 255
        else:
            img1[i, j]= 0
# Display the image
cv2.imshow('sliced image', img1)
cv2.waitKey(0)