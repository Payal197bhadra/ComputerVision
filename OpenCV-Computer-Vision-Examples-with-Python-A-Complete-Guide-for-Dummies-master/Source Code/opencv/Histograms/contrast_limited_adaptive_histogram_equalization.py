import numpy as np
import cv2
import matplotlib.pyplot as plt
# Load the image in greyscale
img = cv2.imread('obama.jpg', 0)
img=cv2.resize(img, (512,512))
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
out = clahe.apply(img)

# Display the images side by side using cv2.hconcat
out1 = cv2.hconcat([img, out])
cv2.imshow('a', out1)
cv2.waitKey(0)
plt.figure()
plt.title("Clahe Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
hist = cv2.calcHist(out, [0], None, [256], [0, 256])
# features.extend(hist)
# plot the histogram
plt.plot(hist, color = 'r')
plt.xlim([0, 256])
plt.show()

#========================================================================
#color image
# Additionally, we can also apply CLAHE to color images, very similar to the
# approach commented in the previous section for the contrast equalization of
# color images, where the results after equalizing only the luminance channel of an
# HSV image are much better than equalizing all the channels of the BGR image.
# In this section, we are going to create four functions in order to equalize the
# color images by using CLAHE only on the luminance channel of different color
# spaces:


def equalize_clahe_color_hsv(img):
    """Equalize the image splitting after conversion to HSV and applying CLAHE
    to the V channel and merging the channels and convert back to BGR
    """
    cla = cv2.createCLAHE(clipLimit=4.0)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cla.apply(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image
def equalize_clahe_color_lab(img):
    """Equalize the image splitting after conversion to LAB and applying CLAHE
    to the L channel and merging the channels and convert back to BGR
    """
    cla = cv2.createCLAHE(clipLimit=4.0)
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    eq_L = cla.apply(L)
    eq_image = cv2.cvtColor(cv2.merge([eq_L, a, b]), cv2.COLOR_Lab2BGR)
    return eq_image
def equalize_clahe_color_yuv(img):
    """Equalize the image splitting after conversion to YUV and applying CLAHE
    to the Y channel and merging the channels and convert back to BGR
    """
    cla = cv2.createCLAHE(clipLimit=4.0)
    Y, U, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eq_Y = cla.apply(Y)
    eq_image = cv2.cvtColor(cv2.merge([eq_Y, U, V]), cv2.COLOR_YUV2BGR)
    return eq_image
def equalize_clahe_color(img):
    """Equalize the image splitting the image applying CLAHE to each channel and merging the results"""
    cla = cv2.createCLAHE(clipLimit=4.0)
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cla.apply(ch))
        eq_image = cv2.merge(eq_channels)
        return eq_image

img_color=cv2.imread("obama.jpg")
img_color=cv2.resize(img_color,(200,200))
equalize_clahe_color_img=equalize_clahe_color(img_color)
equalize_clahe_color_lab_img=equalize_clahe_color_lab(img_color)
equalize_clahe_color_yuv_img=equalize_clahe_color_yuv(img_color)
equalize_clahe_color_hsv_img=equalize_clahe_color_hsv(img_color)
# result=np.hstack((img_color, equalize_clahe_color_img, equalize_clahe_color_lab_img, equalize_clahe_color_yuv_img, equalize_clahe_color_hsv_img))
cv2.imshow("Original Image",img_color)
cv2.imshow("Clahe_color_image",equalize_clahe_color_img)
cv2.imshow("Clahe_color_lab_image", equalize_clahe_color_lab_img)
cv2.imshow("Clahe_color_yuv_image", equalize_clahe_color_yuv_img)
cv2.imshow("Clahe_color_hsv_image", equalize_clahe_color_hsv_img)
cv2.waitKey(0)