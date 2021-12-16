import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("obama.jpg",0)
img=cv2.resize(img,(512,512))
cv2.imshow("Original Image",img)

#flatten the image to get the flattened array
f1=img.flatten()

#calculate the histogram
hist, bins = np.histogram(img, 256, [0,255])

#calculate the cdf cumulative sum
cdf=hist.cumsum()

#places where cdf=0 are ignored and stored as cdf_m
cdf_m=np.ma.masked_equal(cdf,0)

#normalize the cdf to plot with the histogram
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(f1,256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

#iplementing histogram eqaualization transformation with following the formulae
num_cdf_m=(cdf_m - cdf_m.min())*255
den_cdf_m=(cdf_m.max()-cdf_m.min())
cdf_m=num_cdf_m/den_cdf_m

#the masked places of cdf_m are now 0
cdf=np.ma.filled(cdf_m,0).astype('uint8')

#cdf values are assigned in the flattend array
img2=cdf[f1]

#converting 1D image to 2D image to make it original shape
img3=np.reshape(img2, img.shape)
cv2.imshow("Equalized Image",img3)

#calculate the histogram of new image
hist2, bins = np.histogram(img2, 256, [0,255])

#calculate the cdf of equalized histogram
cdf=hist2.cumsum()

#normalize the cdf to plot the histogram
cdf_normalized2 = cdf * float(hist2.max()) / cdf.max()
plt.plot(cdf_normalized2, color = 'b')
plt.hist(img2,256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()


#============================================
#using opencv
#=============================================
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))
cv2.imshow("Equalization", res)


def equalize_hist_color(img):
    """Equalize the image splitting the image applying cv2.equalizeHist() to each channel and merging the results"""
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
        eq_image = cv2.merge(eq_channels)
    return eq_image
img_color=cv2.imread("obama.jpg")
chans = cv2.split(img_color)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
for (chan, color) in zip(chans, colors):
	# create a histogram for the current channel and
	# concatenate the resulting histograms for each
	# channel
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	features.extend(hist)
	# plot the histogram
	plt.plot(hist, color = color)
	plt.xlim([0, 256])
plt.show()
img_color=cv2.resize(img_color,(512,512))
# img_color=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
color_equalized_image=equalize_hist_color(img_color)
chans = cv2.split(color_equalized_image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
for (chan, color) in zip(chans, colors):
	# create a histogram for the current channel and
	# concatenate the resulting histograms for each
	# channel
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	features.extend(hist)
	# plot the histogram
	plt.plot(hist, color = color)
	plt.xlim([0, 256])
plt.show()
result=np.hstack((img_color,color_equalized_image))
cv2.imshow("Equalized colour image",result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#==============================================================================
# We have commented that equalizing the three channels is not a good approach
# because the color shade changes dramatically. This is due to the additive
# properties of the BGR color space. As we are changing both the brightness and
# the contrast in the three channels independently, this can lead to new color
# shades appearing in the image when merging the equalized channels. This issue
# can be seen in the previous screenshot.
# A better approach is to convert the BGR image to a color space containing a
# luminance/intensity channel (Yuv, Lab, HSV, and HSL). Then, we apply
# histogram equalization only on the luminance channel and, finally, perform
# inverse transformation, that is, we merge the channels and convert them back to
# the BGR color space.

def equalize_hist_color_hsv(img_color):
    """Equalize the image splitting the image after HSV conversion and applying cv2.equalizeHist()
    to the V channel, merging the channels and convert back to the BGR color space
    """
    H, S, V = cv2.split(cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image
hsv_equalized_image=equalize_hist_color_hsv(img_color)
result_hsv=np.hstack((result, hsv_equalized_image))
cv2.imshow("HSV_Histogram_Equalization", result_hsv)
cv2.waitKey(0)

# As can be seen, obtained the results obtained after equalizing only the V channel
# of the HSV image are much better than equalizing all the channels of the BGR
# image. As we commented, this approach is also valid for a color space
# containing a luminance/intensity channel (Yuv, Lab, HSV, and HSL). This will
# be seen in the next section.
