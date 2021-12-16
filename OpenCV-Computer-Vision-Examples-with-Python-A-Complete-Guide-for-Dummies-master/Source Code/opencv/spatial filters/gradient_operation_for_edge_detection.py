import numpy as np
import matplotlib.pyplot as plt
import cv2

img= cv2.imread("sudoku-original.jpg",0)
img=cv2.resize(img,(200,200))
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=5)
addition= sobelx+sobely

images= [img, laplacian, sobelx, sobely,addition]
titles= ["Original Image", "Laplacian Image", "Sobel X", "Sobel Y", "Addition"]

for i in range(5):
    plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


#====================================================================
# Image gradient in horizontal direction Sobel x
#======================================================================

image_X = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)

# cv2.imshow("Sobel image", image_X)
# cv2.waitKey()

#=====================================================================
#image gradient in vertical direction Sobel Y
#======================================================================

image_Y = cv2.Sobel(img, cv2.CV_8UC1,0,1)
# cv2.imshow("Sobel image", image_Y)
# cv2.waitKey()

#======================================================================
#calculating the gradient
#=======================================================================
image_X = cv2.convertScaleAbs(image_X)
image_Y = cv2.convertScaleAbs(image_Y)
sobel = cv2.add(image_X, image_Y)
# cv2.imshow("Sobel - L1 norm", sobel)
# cv2.waitKey()

#========================================================================
#performing the thresholding to point out the edges
#==========================================================================
ret,thresh1 = cv2.threshold(sobel,60,255,cv2.THRESH_BINARY)

#===========================================================================
#scharr operator
#============================================================================
scharrx = cv2.Scharr(img, cv2.CV_64F,1,0)
scharry = cv2.Scharr(img, cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharr = cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

images=np.hstack((img, image_X, image_Y, sobel, thresh1))
cv2.imshow("Gradient Image", images)
cv2.waitKey()

#===========================================================================
#Laplacian of Gaussian
#============================================================================
# Apply Gaussian Blur
blur = cv2.GaussianBlur(img,(3,3),0)
# Apply Laplacian operator in some higher datatype
laplacian = cv2.Laplacian(blur,cv2.CV_64F)

# But this tends to localize the edge towards the brighter side.
laplacian1 = laplacian / laplacian.max()
ret, thresh= cv2.threshold(laplacian1, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('a5', thresh)





def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)

    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1], image[i, j - 1], image[i, j + 1],
                         image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h > 0:
                    positive_count += 1
                elif h < 0:
                    negative_count += 1

            # If both negative and positive values exist in
            # the pixel neighborhood, then that pixel is a
            # potential zero crossing

            z_c = ((negative_count > 0) and (positive_count > 0))

            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i, j] > 0:
                    z_c_image[i, j] = image[i, j] + np.abs(e)
                elif image[i, j] < 0:
                    z_c_image[i, j] = np.abs(image[i, j]) + d

    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image / z_c_image.max() * 255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image
zero_crossing= Zero_crossing(laplacian)
cv2.imshow('a7', laplacian1)
cv2.imshow('a8',zero_crossing)
cv2.waitKey(0)


#==============================================================
#Canny Edge Detection
#=============================================================

canny= cv2.Canny(img, 50, 200)
plt.subplot(1,2,1), plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detection")
plt.xticks([]), plt.yticks([])
plt.show()