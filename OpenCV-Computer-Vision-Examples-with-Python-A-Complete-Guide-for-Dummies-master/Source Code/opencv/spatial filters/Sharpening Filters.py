import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


img =cv2.imread("obama.jpg")
img= cv2.resize(img, (400, 500))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#smoothing the image using the median blurring
gray_image_mf = cv2.medianBlur(img, 1)

# Calculate the Laplacian we here implemnet unsharping mask using laplacian operator
lap = cv2.Laplacian(gray_image_mf,cv2.CV_64F)
# Calculate the sharpened image
sharp = img - 0.7*lap


def unsharp(image, sigma, strength):
    # Median filtering
    image_mf = cv2.medianBlur(image, sigma)
    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf, cv2.CV_64F)
    # Calculate the sharpened image
    sharp = image - strength * lap
    # Saturate the pixels in either direction
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0

    return sharp

original_image = plt.imread('obama.jpg')
original_image= cv2.resize(original_image, (400, 500))
sharp1 = np.zeros_like(original_image)
for i in range(3):
    sharp1[:,:,i] = unsharp(original_image[:,:,i], 5, 0.8)
img2=cv2.imread("coin.jpg",0)

#sharpening using laplacian where we take the laplacian kernel and convolove to make the mask and substract
#the mask from orgial image to sharpen the image
kernel = np.array([[0 , 1 , 0] , [1 , -4 , 1] ,[0 , 1 , 0]])
mask= cv2.filter2D(img2,-1, kernel)
sharpen_image = img2 - 1*(mask)


images=[img, gray_image_mf, lap, sharp, original_image,sharp1, sharpen_image]
titles=["image", "Smoothed Image", "Laplacian", "Sharpend Image", "Original Image","Sharpened image","Laplacian sharpend image"]

for i in range(7):
    plt.subplot(3,3,i+1), plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

laplacian_sharpen= np.hstack((img2, sharpen_image))
cv2.imshow("Laplacian sharpend image", laplacian_sharpen)
cv2.waitKey(0)



