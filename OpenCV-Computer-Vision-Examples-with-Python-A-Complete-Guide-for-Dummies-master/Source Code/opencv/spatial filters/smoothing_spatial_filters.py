import cv2
import numpy as np
img=cv2.imread("example-01.png")
img=cv2.resize(img, (300,300))


#==============================================
# 2D Convolution for smoothing image
#==============================================
#creating an average filter kernel
kernel=np.ones((5,5), np.float32)/25
print(kernel)
dst=cv2.filter2D(img, -1, kernel)

#==============================================
#Averaging using box filter or average filter. We can use cv2.blur or cv2.boxFilter() to implement blurring
#==============================================
blur=cv2.blur(img, (5,5))

#==============================================
#Gaussian blur used to remove gaussian noise in the image .It contains sigma x and sigma y value as parameters which
#is standard deviation
#===============================================
gaussian_blur=cv2.GaussianBlur(img, (5,5), 0,0)
img2=np.vstack((img,dst))
img3=np.vstack((blur,gaussian_blur))
img1= np.hstack((
    img2,img3
))

#===============================================
#Medain filter to remove salt and paper noise aling with smoothing of the noise where kernel size must be positive
#odd integer
#===============================================
img4=cv2.imread("example-02.png")
median_blur=cv2.medianBlur(img4,5)


#================================================
#bilateral filtering for smoothing images by removing noises but it preserves the edges
#================================================
bilateral_filter = cv2.bilateralFilter(img,9,75,75)
img6=np.hstack((img,bilateral_filter))
img5=np.hstack((img4,median_blur))
cv2.imshow("image",img1)
cv2.imshow("Median Filter", img5)
cv2.imshow("Bilateral Filter", img6)

cv2.waitKey(0)
cv2.destroyAllWindows()